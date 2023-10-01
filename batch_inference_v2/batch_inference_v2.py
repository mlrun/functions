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
#


from typing import Any, Dict, List, Union

import mlrun

try:
    import mlrun.model_monitoring.api
except ModuleNotFoundError:
    raise mlrun.errors.MLRunNotFoundError(
        f"Please update your `mlrun` version to >=1.5.0 or use an "
        f"older version of the batch inference function."
    )


import numpy as np
import pandas as pd
from mlrun.frameworks.auto_mlrun import AutoMLRun


def _prepare_result_set(
    x: pd.DataFrame, label_columns: List[str], y_pred: np.ndarray
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


def infer(
    context: mlrun.MLClientCtx,
    dataset: Union[mlrun.DataItem, list, dict, pd.DataFrame, pd.Series, np.ndarray],
    model_path: Union[str, mlrun.DataItem],
    drop_columns: Union[str, List[str], int, List[int]] = None,
    label_columns: Union[str, List[str]] = None,
    feature_columns: Union[str, List[str]] = None,
    log_result_set: bool = True,
    result_set_name: str = "prediction",
    batch_id: str = None,
    artifacts_tag: str = "",
    # Drift analysis parameters
    perform_drift_analysis: bool = None,
    trigger_monitoring_job: bool = False,
    batch_image_job: str = "mlrun/mlrun",
    endpoint_id: str = "",
    # The following model endpoint parameters are relevant only if:
    # perform drift analysis is not disabled
    # a new model endpoint record is going to be generated
    model_endpoint_name: str = "batch-infer",
    model_endpoint_drift_threshold: float = 0.7,
    model_endpoint_possible_drift_threshold: float = 0.5,
    model_endpoint_sample_set: Union[
        mlrun.DataItem, list, dict, pd.DataFrame, pd.Series, np.ndarray
    ] = None,
    **predict_kwargs: Dict[str, Any],
):
    """
    Perform a prediction on a given dataset with the given model.
    Can perform drift analysis between the sample set statistics stored in the model to the current input data. The
    drift rule is the value per-feature mean of the TVD and Hellinger scores according to the thresholds configures
    here. When performing drift analysis, this function either uses an existing model endpoint record or creates
    a new one.
    At the moment, this function is supported for `mlrun>=1.5.0` versions.

    :param context:                                 MLRun context.
    :param dataset:                                 The dataset to infer through the model. Can be passed in `inputs` as either a
                                                    Dataset artifact / Feature vector URI. Or, in `parameters` as a list, dictionary or
                                                    numpy array.
    :param model_path:                              The model Store path. Can be provided as an input (DataItem) or as a parameter (string).
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
    :param artifacts_tag:                           Tag to use for all the artifacts resulted from the function (result set and
                                                    model monitoring artifacts)
    :param perform_drift_analysis:                  Whether to perform drift analysis between the sample set of the model object to the
                                                    dataset given. By default, None, which means it will perform drift analysis if the
                                                    model already has feature stats that are considered as a reference sample set.
                                                    Performing drift analysis on a new endpoint id will generate a new model endpoint
                                                    record. Please note that in order to trigger the drift analysis job, you need to
                                                    set `trigger_monitoring_job=True`. Otherwise, the drift analysis will be triggered
                                                    only as part the scheduled monitoring job (if exist in the current project) or
                                                    if triggered manually by the user.
    :param trigger_monitoring_job:                  Whether to trigger the batch drift analysis after the infer job.
    :param batch_image_job:                         The image that will be used to register the monitoring batch job if not exist.
                                                    By default, the image is mlrun/mlrun.
    :param endpoint_id:                             Model endpoint unique ID. If `perform_drift_analysis` was set, the endpoint_id
                                                    will be used either to perform the analysis on existing model endpoint or to
                                                    generate a new model endpoint record.
    :param model_endpoint_name:                     If a new model endpoint is generated, the model name will be presented under this
                                                    endpoint.
    :param model_endpoint_drift_threshold:          The threshold of which to mark drifts. Defaulted to 0.7.
    :param model_endpoint_possible_drift_threshold: The threshold of which to mark possible drifts. Defaulted to 0.5.
    :param model_endpoint_sample_set:               A sample dataset to give to compare the inputs in the drift analysis.
                                                    Can be provided as an input (DataItem) or as a parameter (e.g. string, list, DataFrame).
                                                    The default chosen sample set will always be the one who is set in the model artifact itself.

    raises MLRunInvalidArgumentError: if both `model_path` and `endpoint_id` are not provided
    """

    # Loading the model:
    context.logger.info(f"Loading model...")
    if isinstance(model_path, mlrun.DataItem):
        model_path = model_path.artifact_url
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
    context.logger.info(f"Loading data...")
    x, label_columns = mlrun.model_monitoring.api.read_dataset_as_dataframe(
        dataset=dataset,
        feature_columns=feature_columns,
        label_columns=label_columns,
        drop_columns=drop_columns,
    )

    # Predict:
    context.logger.info(f"Calculating prediction...")
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
        sample_set_statistics = mlrun.model_monitoring.api.get_sample_set_statistics(
            sample_set=model_endpoint_sample_set,
            model_artifact_feature_stats=model_handler._model_artifact.spec.feature_stats,
        )
        mlrun.model_monitoring.api.record_results(
            project=context.project,
            context=context,
            endpoint_id=endpoint_id,
            model_path=model_path,
            model_endpoint_name=model_endpoint_name,
            infer_results_df=result_set.copy(),
            sample_set_statistics=sample_set_statistics,
            drift_threshold=model_endpoint_drift_threshold,
            possible_drift_threshold=model_endpoint_possible_drift_threshold,
            artifacts_tag=artifacts_tag,
            trigger_monitoring_job=trigger_monitoring_job,
            default_batch_image=batch_image_job,
        )
