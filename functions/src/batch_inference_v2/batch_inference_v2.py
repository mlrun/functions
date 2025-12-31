# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from inspect import signature
from typing import Any

import mlrun

try:
    import mlrun.model_monitoring.api
except ModuleNotFoundError:
    raise mlrun.errors.MLRunNotFoundError(
        "Please update your `mlrun` version to >=1.5.0 or use an "
        "older version of the batch inference function."
    )

import numpy as np
import pandas as pd
from mlrun.frameworks.auto_mlrun import AutoMLRun


def _prepare_result_set(
    x: pd.DataFrame, label_columns: list[str], y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Set default label column names and validate given names to prepare the result set - a concatenation of the inputs
    (x) and the model predictions (y_pred).

    :param x:             The inputs.
    :param label_columns: A list of strings representing the target column names to add to the predictions. Default name
                          will be used in case the list is empty (predicted_label_{i}).
    :param y_pred:        The model predictions on the inputs.

    :returns: The result set.

    raises MLRunInvalidArgumentError: If the labels columns amount do not match the outputs or if one of the label
                                       column already exists in the dataset.
    """
    # Prepare default target columns names if not provided:
    prediction_columns_amount = 1 if len(y_pred.shape) == 1 else y_pred.shape[1]
    if len(label_columns) == 0:
        # Add default label column names:
        if prediction_columns_amount == 1:
            label_columns = ["predicted_label"]
        else:
            label_columns = [
                f"predicted_label_{i}" for i in range(prediction_columns_amount)
            ]

    # Validate the label columns:
    if prediction_columns_amount != len(label_columns):
        # No equality between provided label column names and outputs amount:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The number of predicted labels: {prediction_columns_amount} "
            f"is not equal to the given label columns: {len(label_columns)}"
        )
    common_labels = set(label_columns) & set(x.columns.tolist())
    if common_labels:
        # Label column exist in the original inputs:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The labels: {common_labels} are already existed in the given dataset."
        )

    return pd.concat(
        [x, pd.DataFrame(y_pred, columns=label_columns, index=x.index)], axis=1
    )


def _get_sample_set_statistics_parameters(
    context: mlrun.MLClientCtx,
    model_endpoint_sample_set: mlrun.DataItem | list | dict | pd.DataFrame | pd.Series | np.ndarray,
    model_artifact_feature_stats: dict,
    feature_columns: list | None,
    drop_columns: list | None,
    label_columns: list | None,
) -> dict[str, Any]:
    statics_input_full_dict = dict(
        sample_set=model_endpoint_sample_set,
        model_artifact_feature_stats=model_artifact_feature_stats,
        sample_set_columns=feature_columns,
        sample_set_drop_columns=drop_columns,
        sample_set_label_columns=label_columns,
    )
    get_sample_statics_function = mlrun.model_monitoring.api.get_sample_set_statistics
    statics_function_input_dict = signature(get_sample_statics_function).parameters
    #  As a result of changes to input parameters in the mlrun-get_sample_set_statistics function,
    #  we will now send only the parameters it expects.
    statistics_input_filtered = {
        key: statics_input_full_dict[key] for key in statics_function_input_dict
    }
    if len(statistics_input_filtered) != len(statics_function_input_dict):
        context.logger.warning(
            f"get_sample_set_statistics is in an older version; "
            "some parameters will not be sent to the function."
            f" Expected input: {list(statics_function_input_dict.keys())},"
            f" actual input: {list(statistics_input_filtered.keys())}"
        )
    return statistics_input_filtered


def infer(
    context: mlrun.MLClientCtx,
    dataset: mlrun.DataItem | list | dict | pd.DataFrame | pd.Series | np.ndarray,
    model_path: str | mlrun.DataItem,
    drop_columns: str | list[str] | int | list[int] = None,
    label_columns: str | list[str] = None,
    feature_columns: str | list[str] = None,
    log_result_set: bool = True,
    result_set_name: str = "prediction",
    batch_id: str = None,
    artifacts_tag: str = "",
    # Drift analysis parameters
    perform_drift_analysis: bool = None,
    endpoint_id: str = "",
    # The following model endpoint parameters are relevant only if:
    # perform drift analysis is not disabled
    # a new model endpoint record is going to be generated
    model_endpoint_name: str = "batch-infer",
    model_endpoint_sample_set: mlrun.DataItem | list | dict | pd.DataFrame | pd.Series | np.ndarray = None,
    # the following parameters are deprecated and will be removed once the versioning mechanism is implemented
    # TODO: Remove the following parameters once FHUB-13 is resolved
    trigger_monitoring_job: bool | None = None,
    batch_image_job: str | None = None,
    model_endpoint_drift_threshold: float | None = None,
    model_endpoint_possible_drift_threshold: float | None = None,
    # prediction kwargs to pass to the model predict function
    **predict_kwargs: dict[str, Any],
):
    """
    Perform a prediction on the provided dataset using the specified model.
    Ensure that the model has already been logged under the current project.

    If you wish to apply monitoring tools (e.g., drift analysis), set the perform_drift_analysis parameter to True.
    This will create a new model endpoint record under the specified model_endpoint_name.
    Additionally, ensure that model monitoring is enabled at the project level by calling the
    project.enable_model_monitoring() function. You can also apply monitoring to an existing model by providing its
    endpoint id or name, and the monitoring tools will be applied to that endpoint.

    At the moment, this function is supported for `mlrun>=1.5.0` versions.

    :param context:                                 MLRun context.
    :param dataset:                                 The dataset to infer through the model. Provided as an input (DataItem)
                                                    that represents Dataset artifact / Feature vector URI.
                                                    If using MLRun SDK, `dataset` can also be provided as a list, dictionary or
                                                    numpy array.
    :param model_path:                              Model store uri (should start with store://). Provided as an input (DataItem).
                                                    If using MLRun SDK, `model_path` can also be provided as a parameter (string).
                                                    To generate a valid model store URI, please log the model before running this function.
                                                    If `endpoint_id` of existing model endpoint is provided, make sure
                                                    that it has a similar model store path, otherwise the drift analysis
                                                    won't be triggered.
    :param drop_columns:                            A string / integer or a list of strings / integers that represent the column names
                                                    / indices to drop. When the dataset is a list or a numpy array this parameter must
                                                    be represented by integers.
    :param label_columns:                           The target label(s) of the column(s) in the dataset for Regression or
                                                    Classification tasks. The label column can be accessed from the model object, or
                                                    the feature vector provided if available.
    :param feature_columns:                         List of feature columns that will be used to build the dataframe when dataset is
                                                    from type list or numpy array.
    :param log_result_set:                          Whether to log the result set - a DataFrame of the given inputs concatenated with
                                                    the predictions. Defaulted to True.
    :param result_set_name:                         The db key to set name of the prediction result and the filename. Defaulted to
                                                    'prediction'.
    :param batch_id:                                The ID of the given batch (inference dataset). If `None`, it will be generated.
                                                    Will be logged as a result of the run.
    :param artifacts_tag:                           Tag to use for prediction set result artifact.
    :param perform_drift_analysis:                  Whether to perform drift analysis between the sample set of the model object to the
                                                    dataset given. By default, None, which means it will perform drift analysis if the
                                                    model already has feature stats that are considered as a reference sample set.
                                                    Performing drift analysis on a new endpoint id will generate a new model endpoint
                                                    record.
    :param endpoint_id:                             Model endpoint unique ID. If `perform_drift_analysis` was set, the endpoint_id
                                                    will be used to perform the analysis on existing model endpoint, or if it does not
                                                    exist a new model endpoint will be created with a newly generated ID.
    :param model_endpoint_name:                     If a new model endpoint is generated, the model name will be presented under this
                                                    endpoint.
    :param model_endpoint_sample_set:               A sample dataset to give to compare the inputs in the drift analysis.
                                                    Can be provided as an input (DataItem) or as a parameter (e.g. string, list, DataFrame).
                                                    The default chosen sample set will always be the one who is set in the model artifact itself.
    :param trigger_monitoring_job:                  Whether to trigger the batch drift analysis after the infer job.
    :param batch_image_job:                         The image that will be used to register the monitoring batch job if not exist.
                                                    By default, the image is mlrun/mlrun.
    :param model_endpoint_drift_threshold:          The threshold of which to mark drifts. Defaulted to 0.7.
    :param model_endpoint_possible_drift_threshold: The threshold of which to mark possible drifts. Defaulted to 0.5.

    raises MLRunInvalidArgumentError: if both `model_path` and `endpoint_id` are not provided
    """

    if trigger_monitoring_job:
        context.logger.warning(
            "The `trigger_monitoring_job` parameter is deprecated and will be removed once the versioning mechanism is implemented. "
            "if you are using mlrun<1.7.0, please import the previous version of this function, for example "
            "'hub://batch_inference_v2:2.5.0'."
        )
    if batch_image_job:
        context.logger.warning(
            "The `batch_image_job` parameter is deprecated and will be removed once the versioning mechanism is implemented. "
            "if you are using mlrun<1.7.0, please import the previous version of this function, for example "
            "'hub://batch_inference_v2:2.5.0'."
        )
    if model_endpoint_drift_threshold:
        context.logger.warning(
            "The `model_endpoint_drift_threshold` parameter is deprecated and will be removed once the versioning mechanism is implemented. "
            "if you are using mlrun<1.7.0, please import the previous version of this function, for example "
            "'hub://batch_inference_v2:2.5.0'."
        )
    if model_endpoint_possible_drift_threshold:
        context.logger.warning(
            "The `model_endpoint_possible_drift_threshold` parameter is deprecated and will be removed once the versioning mechanism is implemented. "
            "if you are using mlrun<1.7.0, please import the previous version of this function, for example "
            "'hub://batch_inference_v2:2.5.0'."
        )

    # Loading the model:
    context.logger.info("Loading model...")
    if isinstance(model_path, mlrun.DataItem):
        model_path = model_path.artifact_url
    if not mlrun.datastore.is_store_uri(model_path):
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The provided model path ({model_path}) is invalid - should start with `store://`. "
            f"Please make sure that you have logged the model using `project.log_model()` "
            f"which generates a unique store uri for the logged model."
        )
    model_handler = AutoMLRun.load_model(model_path=model_path, context=context)

    if label_columns is None:
        label_columns = [
            output.name for output in model_handler._model_artifact.spec.outputs
        ]

    if feature_columns is None:
        feature_columns = [
            input.name for input in model_handler._model_artifact.spec.inputs
        ]

    # Get dataset by object, URL or by FeatureVector:
    context.logger.info("Loading data...")
    x, label_columns = mlrun.model_monitoring.api.read_dataset_as_dataframe(
        dataset=dataset,
        feature_columns=feature_columns,
        label_columns=label_columns,
        drop_columns=drop_columns,
    )

    # Predict:
    context.logger.info("Calculating prediction...")
    y_pred = model_handler.model.predict(x, **predict_kwargs)

    # Prepare the result set:
    result_set = _prepare_result_set(x=x, label_columns=label_columns, y_pred=y_pred)

    # Check for logging the result set:
    if log_result_set:
        mlrun.model_monitoring.api.log_result(
            context=context,
            result_set_name=result_set_name,
            result_set=result_set,
            artifacts_tag=artifacts_tag,
            batch_id=batch_id,
        )

    # Check for performing drift analysis
    if (
        perform_drift_analysis is None
        and model_handler._model_artifact.spec.feature_stats is not None
    ):
        perform_drift_analysis = True
    if perform_drift_analysis:
        context.logger.info("Performing drift analysis...")
        # Get the sample set statistics (either from the sample set or from the statistics logged with the model)
        statistics_input_filtered = _get_sample_set_statistics_parameters(
            context=context,
            model_endpoint_sample_set=model_endpoint_sample_set,
            model_artifact_feature_stats=model_handler._model_artifact.spec.feature_stats,
            feature_columns=feature_columns,
            drop_columns=drop_columns,
            label_columns=label_columns,
        )
        sample_set_statistics = mlrun.model_monitoring.api.get_sample_set_statistics(
            **statistics_input_filtered
        )
        mlrun.model_monitoring.api.record_results(
            project=context.project,
            context=context,
            endpoint_id=endpoint_id,
            model_path=model_path,
            model_endpoint_name=model_endpoint_name,
            infer_results_df=result_set.copy(),
            sample_set_statistics=sample_set_statistics,
        )
