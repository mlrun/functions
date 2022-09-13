# Copyright 2019 Iguazio
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
import json
from typing import Any, Dict, List, Tuple, Union

import mlrun
import numpy as np
import pandas as pd
from mlrun import feature_store as fs
from mlrun.api.schemas import ObjectKind
from mlrun.artifacts import Artifact
from mlrun.data_types.infer import InferOptions, get_df_stats
from mlrun.frameworks.auto_mlrun import AutoMLRun
from mlrun.model_monitoring.features_drift_table import FeaturesDriftTablePlot
from mlrun.model_monitoring.model_monitoring_batch import (
    VirtualDrift,
    calculate_inputs_statistics,
)

DatasetType = Union[mlrun.DataItem, list, dict, pd.DataFrame, pd.Series, np.ndarray]


def _read_dataset_as_dataframe(
    dataset: DatasetType,
    label_columns: Union[str, List[str]] = None,
    drop_columns: Union[str, List[str], int, List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse the given dataset into a DataFrame and drop the columns accordingly. In addition, the label columns will be
    parsed and validated as well.

    :param dataset:       The dataset to train the model on.
                          Can be either a list of lists, dict, URI or a FeatureVector.
    :param label_columns: The target label(s) of the column(s) in the dataset. for Regression or
                          Classification tasks.
    :param drop_columns:  ``str`` / ``int`` or a list of ``str`` / ``int`` that represent the column names / indices to
                          drop.

    :returns: A tuple of:
              [0] = The parsed dataset as a DataFrame
              [1] = Label columns.

    raises MLRunInvalidArgumentError: If the `drop_columns` are not matching the dataset or unsupported dataset type.
    """
    # Turn the `drop labels` into a list if given:
    if drop_columns is not None:
        if not isinstance(drop_columns, list):
            drop_columns = [drop_columns]

    # Check if the dataset is in fact a Feature Vector:
    if dataset.meta and dataset.meta.kind == ObjectKind.feature_vector:
        # Try to get the label columns if not provided:
        label_columns = label_columns or dataset.meta.status.label_column
        # Get the features and parse to DataFrame:
        dataset = fs.get_offline_features(
            dataset.meta.uri, drop_columns=drop_columns
        ).to_dataframe()
    else:
        # Parse to DataFrame according to the dataset's type:
        if isinstance(dataset, (list, np.ndarray)):
            # Parse the list / numpy array into a DataFrame:
            dataset = pd.DataFrame(dataset)
            # Validate the `drop_columns` is given as integers:
            if drop_columns and not all(isinstance(col, int) for col in drop_columns):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "`drop_columns` must be an integer / list of integers if provided as a list."
                )
        elif isinstance(dataset, mlrun.DataItem):
            # Turn the DataITem to DataFrame:
            dataset = dataset.as_df()
        else:
            # Parse the object (should be a pd.DataFrame / pd.Series, dictionary) into a DataFrame:
            try:
                dataset = pd.DataFrame(dataset)
            except ValueError as e:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Could not parse the given dataset of type {type(dataset)} into a pandas DataFrame. "
                    f"Received the following error: {e}"
                )
        # Drop columns if needed:
        if drop_columns:
            dataset.drop(drop_columns, axis=1, inplace=True)

    # Turn the `label_columns` into a list by default:
    if label_columns is None:
        label_columns = []
    elif isinstance(label_columns, (str, int)):
        label_columns = [label_columns]

    return dataset, label_columns


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

    return pd.concat([x, pd.DataFrame(y_pred, columns=label_columns)], axis=1)


def _get_sample_set_statistics(
    sample_set: DatasetType = None, model_artifact_feature_stats: dict = None
) -> dict:
    """
    Get the sample set statistics either from the given sample set or the statistics logged with the model while
    favoring the given sample set.

    :param sample_set:                   A sample dataset to give to compare the inputs in the drift analysis.
    :param model_artifact_feature_stats: The `feature_stats` attribute in the spec of the model artifact, where the
                                         original sample set statistics of the model was used.

    :returns: The sample set statistics.

    raises MLRunInvalidArgumentError: If no sample set or statistics were given.
    """
    # Check if a sample set was provided:
    if sample_set is None:
        # Check if the model was logged with a sample set:
        if model_artifact_feature_stats is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Cannot perform drift analysis as there is no sample set to compare to. The model artifact was not "
                "logged with a sample set and `sample_set` was not provided to the function."
            )
        # Return the statistics logged with the model:
        return model_artifact_feature_stats

    # Return the sample set statistics:
    return get_df_stats(df=sample_set, options=InferOptions.Histogram)


def _get_drift_result(
    tvd: float,
    hellinger: float,
    threshold: float,
) -> Tuple[bool, float]:
    """
    Calculate the drift result by the following equation: (tvd + hellinger) / 2

    :param tvd:       The feature's TVD value.
    :param hellinger: The feature's Hellinger value.
    :param threshold: The threshold from which the value is considered a drift.

    :returns: A tuple of:
              [0] = Boolean value as the drift status.
              [1] = The result.
    """
    result = (tvd + hellinger) / 2
    if result >= threshold:
        return True, result
    return False, result


def _perform_drift_analysis(
    sample_set_statistics: dict,
    inputs: pd.DataFrame,
    drift_threshold: float,
    possible_drift_threshold: float,
    inf_capping: float,
) -> Tuple[Artifact, Artifact, dict]:
    """
    Perform drift analysis, producing the drift table artifact for logging post prediction.

    :param sample_set_statistics:    The statistics of the sample set logged along a model.
    :param inputs:                   Input dataset to perform the drift calculation on.
    :param drift_threshold:          The threshold of which to mark drifts.
    :param possible_drift_threshold: The threshold of which to mark possible drifts.
    :param inf_capping:              The value to set for when it reached infinity.

    :returns: A tuple of
              [0] = An MLRun artifact holding the HTML code of the drift table plot.
              [1] = An MLRun artifact holding the metric per feature dictionary.
              [2] = Results to log the final analysis outcome.
    """
    # Calculate the input's statistics:
    inputs_statistics = calculate_inputs_statistics(
        sample_set_statistics=sample_set_statistics,
        inputs=inputs,
    )

    # Calculate drift:
    virtual_drift = VirtualDrift(inf_capping=inf_capping)
    metrics = virtual_drift.compute_drift_from_histograms(
        feature_stats=sample_set_statistics,
        current_stats=inputs_statistics,
    )
    drift_results = virtual_drift.check_for_drift_per_feature(
        metrics_results_dictionary=metrics,
        possible_drift_threshold=possible_drift_threshold,
        drift_detected_threshold=drift_threshold,
    )

    # Validate all feature columns named the same between the inputs and sample sets:
    sample_features = set(
        [
            feature_name
            for feature_name, feature_statistics in sample_set_statistics.items()
            if isinstance(feature_statistics, dict)
        ]
    )
    input_features = set(inputs.columns)
    if len(sample_features & input_features) != len(input_features):
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Not all feature names were matching between the inputs and the sample set provided: "
            f"{input_features - sample_features | sample_features - input_features}"
        )

    # Plot:
    html_plot = FeaturesDriftTablePlot().produce(
        features=list(input_features),
        sample_set_statistics=sample_set_statistics,
        inputs_statistics=inputs_statistics,
        metrics=metrics,
        drift_results=drift_results,
    )

    # Prepare metrics per feature dictionary:
    metrics_per_feature = {
        feature: _get_drift_result(
            tvd=metric_dictionary["tvd"],
            hellinger=metric_dictionary["hellinger"],
            threshold=drift_threshold,
        )[1]
        for feature, metric_dictionary in metrics.items()
        if isinstance(metric_dictionary, dict)
    }

    # Calculate the final analysis result:
    drift_status, drift_metric = _get_drift_result(
        tvd=metrics["tvd_mean"],
        hellinger=metrics["hellinger_mean"],
        threshold=drift_threshold,
    )

    return (
        Artifact(body=html_plot, format="html", key="drift_table_plot"),
        Artifact(
            body=json.dumps(metrics_per_feature),
            format="json",
            key="features_drift_results",
        ),
        {"drift_status": drift_status, "drift_metric": drift_metric},
    )


def predict(
    context: mlrun.MLClientCtx,
    model: str,
    dataset: DatasetType,
    drop_columns: Union[str, List[str], int, List[int]] = None,
    label_columns: Union[str, List[str]] = None,
    log_result_set: bool = True,
    result_set_name: str = "prediction",
    perform_drift_analysis: bool = None,
    sample_set: DatasetType = None,
    drift_threshold: float = 0.7,
    possible_drift_threshold: float = 0.5,
    inf_capping: float = 10.0,
    artifacts_tag: str = "",
    **predict_kwargs: Dict[str, Any],
):
    """
    Perform a prediction on a given dataset with the given model. Can perform drift analysis between the sample set
    statistics stored in the model to the current input data. The drift rule is the value per-feature mean of the TVD
    and Hellinger scores according to the thresholds configures here.

    :param context:                  MLRun context.
    :param model:                    The model Store path.
    :param dataset:                  The dataset to infer through the model. Can be passed in `inputs` as either a
                                     Dataset artifact / Feature vector URI. Or, in `parameters` as a list, dictionary or
                                     numpy array.
    :param drop_columns:             A string / integer or a list of strings / integers that represent the column names
                                     / indices to drop. When the dataset is a list or a numpy array this parameter must
                                     be represented by integers.
    :param label_columns:            The target label(s) of the column(s) in the dataset for Regression or
                                     Classification tasks.
    :param log_result_set:           Whether to log the result set - a DataFrame of the given inputs concatenated with
                                     the predictions. Defaulted to True.
    :param result_set_name:          The db key to set name of the prediction result and the filename. Defaulted to
                                     'prediction'.
    :param perform_drift_analysis:   Whether to perform drift analysis between the sample set of the model object to the
                                     dataset given. By default, None, which means it will perform drift analysis if the
                                     model has a sample set statistics. Perform drift analysis will produce a data drift
                                     table artifact.
    :param sample_set:               A sample dataset to give to compare the inputs in the drift analysis. The default
                                     chosen sample set will always be the one who is set in the model artifact itself.
    :param drift_threshold:          The threshold of which to mark drifts. Defaulted to 0.7.
    :param possible_drift_threshold: The threshold of which to mark possible drifts. Defaulted to 0.5.
    :param inf_capping:              The value to set for when it reached infinity. Defaulted to 10.0.
    :param artifacts_tag:            Tag to use for all the artifacts resulted from the function.
    """
    # Get dataset by URL or by FeatureVector:
    x, label_columns = _read_dataset_as_dataframe(
        dataset=dataset,
        label_columns=label_columns,
        drop_columns=drop_columns,
    )

    # Loading the model:
    context.logger.info(f"Loading model...")
    model_handler = AutoMLRun.load_model(model_path=model, context=context)

    # Predict:
    context.logger.info(f"Calculating prediction...")
    y_pred = model_handler.model.predict(x, **predict_kwargs)

    # Prepare the result set:
    result_set = _prepare_result_set(x=x, label_columns=label_columns, y_pred=y_pred)

    # Check for logging the result set:
    if log_result_set:
        context.logger.info(f"Logging result set (x | prediction)...")
        context.log_dataset(
            key=result_set_name,
            df=result_set,
            db_key=result_set_name,
            tag=artifacts_tag,
        )

    # Check for performing drift analysis:
    if (
        perform_drift_analysis is None
        and model_handler._model_artifact.spec.feature_stats is not None
    ):
        perform_drift_analysis = True
    if perform_drift_analysis:
        context.logger.info("Performing drift analysis...")
        # Get the sample set statistics (either from the sample set or from the statistics logged with the model):
        sample_set_statistics = _get_sample_set_statistics(
            sample_set=sample_set,
            model_artifact_feature_stats=model_handler._model_artifact.spec.feature_stats,
        )
        # Produce the artifact:
        (
            drift_table_plot,
            metric_per_feature_dict,
            analysis_results,
        ) = _perform_drift_analysis(
            sample_set_statistics=sample_set_statistics,
            inputs=result_set,
            drift_threshold=drift_threshold,
            possible_drift_threshold=possible_drift_threshold,
            inf_capping=inf_capping,
        )
        # Log the artifact and results:
        context.log_artifact(drift_table_plot, tag=artifacts_tag)
        context.log_artifact(metric_per_feature_dict, tag=artifacts_tag)
        context.log_results(results=analysis_results)
