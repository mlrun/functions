from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import json
import mlrun
import pandas as pd
from sklearn.model_selection import train_test_split

from mlrun.artifacts import Artifact
from mlrun.frameworks.auto_mlrun import AutoMLRun
from mlrun import feature_store as fs
from mlrun.api.schemas import ObjectKind
from mlrun.utils.helpers import create_class, create_function
from mlrun.model_monitoring.features_drift_table import FeaturesDriftTablePlot
from mlrun.model_monitoring.model_monitoring_batch import (
    VirtualDrift,
    calculate_inputs_statistics,
)

PathType = Union[str, Path]


class KWArgsPrefixes:
    MODEL_CLASS = "CLASS_"
    FIT = "FIT_"
    TRAIN = "TRAIN_"
    PREDICT = "PREDICT_"


def _more_than_one(arg_list: List) -> bool:
    return len([1 for arg in arg_list if arg is not None]) > 1


def _get_sub_dict_by_prefix(src: Dict, prefix_key: str) -> Dict[str, Any]:
    """
    Collect all the keys from the given dict that starts with the given prefix and creates a new dictionary with these
    keys.

    :param src:         The source dict to extract the values from.
    :param prefix_key:  Only keys with this prefix will be returned. The keys in the result dict will be without this
                        prefix.
    """
    return {
        key.replace(prefix_key, ""): val
        for key, val in src.items()
        if key.startswith(prefix_key)
    }


def _get_dataframe(
    context: mlrun.MLClientCtx,
    dataset: mlrun.DataItem,
    label_columns: Optional[Union[str, List[str]]] = None,
    drop_columns: Union[str, List[str], int, List[int]] = None,
) -> Tuple[pd.DataFrame, Optional[Union[str, List[str]]]]:
    """
    Getting the DataFrame of the dataset and drop the columns accordingly.

    :param context:         MLRun context.
    :param dataset:         The dataset to train the model on.
                            Can be either a list of lists, dict, URI or a FeatureVector.
    :param label_columns:   The target label(s) of the column(s) in the dataset. for Regression or
                            Classification tasks.
    :param drop_columns:    str/int or a list of strings/ints that represent the column names/indices to drop.
    """
    if isinstance(dataset, (list, dict)):
        dataset = pd.DataFrame(dataset)
        # Checking if drop_columns provided by integer type:
        if drop_columns:
            if isinstance(drop_columns, str) or (
                isinstance(drop_columns, list)
                and any(isinstance(col, str) for col in drop_columns)
            ):
                context.logger.error(
                    "drop_columns must be an integer/list of integers if not provided with a URI/FeatureVector dataset"
                )
                raise ValueError
            dataset.drop(drop_columns, axis=1, inplace=True)

        return dataset, label_columns

    if dataset.meta and dataset.meta.kind == ObjectKind.feature_vector:
        # feature-vector case:
        label_columns = label_columns or dataset.meta.status.label_column
        dataset = fs.get_offline_features(
            dataset.meta.uri, drop_columns=drop_columns
        ).to_dataframe()

        context.logger.info(f"label columns: {label_columns}")
    else:
        # simple URL case:
        dataset = dataset.as_df()
        if drop_columns:
            if all(col in dataset for col in drop_columns):
                dataset = dataset.drop(drop_columns, axis=1)
            else:
                context.logger.info(
                    "not all of the columns to drop in the dataset, drop columns process skipped"
                )
    return dataset, label_columns


def train(
    context: mlrun.MLClientCtx,
    dataset: mlrun.DataItem,
    drop_columns: List[str] = None,
    model_class: str = None,
    model_name: str = "model",
    tag: str = "",
    label_columns: Optional[Union[str, List[str]]] = None,
    sample_set: mlrun.DataItem = None,
    test_set: mlrun.DataItem = None,
    train_test_split_size: float = None,
    random_state: int = None,
):
    """
    Training the given model on the given dataset.

    :param context:                 MLRun context
    :param dataset:                 The dataset to train the model on. Can be either a URI or a FeatureVector
    :param drop_columns:            str or a list of strings that represent the columns to drop
    :param model_class:             The class of the model, e.g. `sklearn.linear_model.LogisticRegression`
    :param model_name:              The model's name to use for storing the model artifact, default to 'model'
    :param tag:                     The model's tag to log with
    :param label_columns:           The target label(s) of the column(s) in the dataset. for Regression or
                                    Classification tasks
    :param sample_set:              A sample set of inputs for the model for logging its stats along the model in favour
                                    of model monitoring. Can be either a URI or a FeatureVector
    :param test_set:                The test set to train the model with
    :param train_test_split_size:   Should be between 0.0 and 1.0 and represent the proportion of the dataset to include
                                    in the test split. The size of the Training set is set to the complement of this
                                    value. Default = 0.2
    :param random_state:            Random state for `train_test_split`
    """
    # Validate inputs:
    # Check if exactly one of them is supplied:
    if test_set is None:
        if train_test_split_size is None:
            context.logger.info(
                "test_set or train_test_split_size are not provided, setting train_test_split_size to 0.2"
            )
            train_test_split_size = 0.2

    elif train_test_split_size:
        context.logger.info(
            "test_set provided, ignoring given train_test_split_size value"
        )
        train_test_split_size = None

    # Get DataFrame by URL or by FeatureVector:
    dataset, label_columns = _get_dataframe(
        context=context,
        dataset=dataset,
        label_columns=label_columns,
        drop_columns=drop_columns,
    )

    # Getting the sample set:
    if sample_set is None:
        context.logger.info(
            f"Sample set not given, using the whole training set as the sample set"
        )
        sample_set = dataset
    else:
        sample_set, _ = _get_dataframe(
            context=context,
            dataset=sample_set,
            label_columns=label_columns,
            drop_columns=drop_columns,
        )

    # Parsing kwargs:
    # TODO: Use in xgb or lgbm train function.
    train_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.TRAIN
    )
    fit_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.FIT
    )
    model_class_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.MODEL_CLASS
    )

    # Check if model or function:
    if hasattr(model_class, "train"):
        # TODO: Need to call: model(), afterwards to start the train function.
        # model = create_function(f"{model_class}.train")
        raise NotImplementedError
    else:
        # Creating model instance:
        model = create_class(model_class)(**model_class_kwargs)

    x = dataset.drop(label_columns, axis=1)
    y = dataset[label_columns]
    if train_test_split_size:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=train_test_split_size, random_state=random_state
        )
    else:
        x_train, y_train = x, y

        test_set = test_set.as_df()
        if drop_columns:
            test_set = dataset.drop(drop_columns, axis=1)

        x_test, y_test = test_set.drop(label_columns, axis=1), test_set[label_columns]

    AutoMLRun.apply_mlrun(
        model=model,
        model_name=model_name,
        context=context,
        tag=tag,
        sample_set=sample_set,
        y_columns=label_columns,
        test_set=test_set,
        x_test=x_test,
        y_test=y_test,
    )
    context.logger.info(f"training '{model_name}'")
    model.fit(x_train, y_train, **fit_kwargs)


def evaluate(
    context: mlrun.MLClientCtx,
    model: str,
    dataset: mlrun.DataItem,
    drop_columns: List[str] = None,
    label_columns: Optional[Union[str, List[str]]] = None,
):
    """
    Evaluating a model. Artifacts generated by the MLHandler.

    :param context:                 MLRun context.
    :param model:                   The model Store path.
    :param dataset:                 The dataset to evaluate the model on. Can be either a URI or a FeatureVector.
    :param drop_columns:            str or a list of strings that represent the columns to drop.
    :param label_columns:           The target label(s) of the column(s) in the dataset. for Regression or
                                    Classification tasks.
    """
    # Get dataset by URL or by FeatureVector:
    dataset, label_columns = _get_dataframe(
        context=context,
        dataset=dataset,
        label_columns=label_columns,
        drop_columns=drop_columns,
    )

    # Parsing label_columns:
    parsed_label_columns = []
    if label_columns:
        label_columns = (
            label_columns if isinstance(label_columns, list) else [label_columns]
        )
        for lc in label_columns:
            if fs.common.feature_separator in lc:
                feature_set_name, label_name, alias = fs.common.parse_feature_string(lc)
                parsed_label_columns.append(alias or label_name)
        if parsed_label_columns:
            label_columns = parsed_label_columns

    # Parsing kwargs:
    predict_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.PREDICT
    )

    x = dataset.drop(label_columns, axis=1)
    y = dataset[label_columns]

    # Loading the model and predicting:
    model_handler = AutoMLRun.load_model(model_path=model, context=context)
    AutoMLRun.apply_mlrun(model_handler.model, y_test=y, model_path=model)

    context.logger.info(f"evaluating '{model_handler.model_name}'")
    model_handler.model.predict(x, **predict_kwargs)


def _perform_drift_analysis(
    context: mlrun.MLClientCtx,
    sample_set_statistics: dict,
    inputs: pd.DataFrame,
    drift_threshold: float,
    possible_drift_threshold: float,
) -> Tuple[Artifact, Artifact, Artifact]:
    """
    Perform drift analysis, producing the drift table artifact for logging post prediction.

    :param context:                  MLRun context.
    :param sample_set_statistics:    The statistics of the sample set logged along a model.
    :param inputs:                   Input dataset to perform the drift calculation on.
    :param drift_threshold:          The threshold of which to mark drifts. Defaulted to 0.7.
    :param possible_drift_threshold: The threshold of which to mark possible drifts. Defaulted to 0.5.

    :returns: An MLRun artifact holding the HTML code of the drift table plot.
    """
    # Calculate the inputs statistics:
    inputs_statistics = calculate_inputs_statistics(
        sample_set_statistics=sample_set_statistics,
        inputs=inputs,
    )

    # Calculate drift:
    virtual_drift = VirtualDrift(inf_capping=10)
    metrics = virtual_drift.compute_drift_from_histograms(
        feature_stats=sample_set_statistics,
        current_stats=inputs_statistics,
    )
    drift_results = virtual_drift.check_for_drift_per_feature(
        metrics_results_dictionary=metrics,
        possible_drift_threshold=possible_drift_threshold,
        drift_detected_threshold=drift_threshold,
    )

    # Discover common features (in ideal scenario this check should not occur, data should be validated before getting
    # into prediction):
    sample_features = set(
        [
            feature_name
            for feature_name, feature_statistics in sample_set_statistics.items()
            if isinstance(feature_statistics, dict)
        ]
    )
    features = inputs.columns
    if len(sample_features & set(features)) == 0:
        raise ValueError(
            "No matching features (column names) provided "
            "between the sample set in the model and the current inputs."
        )
    elif len(sample_features & set(features)) != len(features):
        context.logger.warn(
            f"Not all feature names were matching between the inputs and the sample set in the model. "
            f"The following features won't be included in the drift analysis: "
            f"{set(features) - sample_features | sample_features - set(features)}"
        )
        features = list(sample_features & set(features))

    # Plot:
    html_plot = FeaturesDriftTablePlot().produce(
        features=features,
        sample_set_statistics=sample_set_statistics,
        inputs_statistics=inputs_statistics,
        metrics=metrics,
        drift_results=drift_results,
    )

    return (
        Artifact(body=html_plot, format="html", key="drift_table_plot"),
        Artifact(body=json.dumps(metrics), format="json", key="drift_results"),
        Artifact(
            body=json.dumps(inputs_statistics), format="json", key="dataset_statistics"
        ),
    )


def predict(
    context: mlrun.MLClientCtx,
    model: str,
    dataset: mlrun.DataItem,
    drop_columns: Union[str, List[str], int, List[int]] = None,
    label_columns: Optional[Union[str, List[str]]] = None,
    perform_drift_analysis: bool = None,
    drift_threshold: float = 0.7,
    possible_drift_threshold: float = 0.5,
):
    """
    Perform a prediction on a given dataset with the given model. Can perform drift analysis between the sample set
    statistics stored in the model to the current input data. The drift rule is the value per-feature mean of the TVD
    and Hellinger scores according to the thresholds configures here.

    :param context:                  MLRun context.
    :param model:                    The model Store path.
    :param dataset:                  The dataset to evaluate the model on. Can be either a URI, a FeatureVector or a
                                     sample in a shape of a list/dict.
    :param drop_columns:             str/int or a list of strings/ints that represent the column names/indices to drop.
                                     When the dataset is a list/dict this parameter should be represented by integers.
    :param label_columns:            The target label(s) of the column(s) in the dataset. for Regression or
                                     Classification tasks.
    :param perform_drift_analysis:   Whether to perform drift analysis between the sample set of the model object to the
                                     dataset given. By default, None, which means it will perform drift analysis if the
                                     model has a sample set statistics. Perform drift analysis will produce a data drift
                                     table artifact.
    :param drift_threshold:          The threshold of which to mark drifts. Defaulted to 0.7.
    :param possible_drift_threshold: The threshold of which to mark possible drifts. Defaulted to 0.5.
    """
    # Get dataset by URL or by FeatureVector:
    dataset, label_columns = _get_dataframe(
        context=context,
        dataset=dataset,
        label_columns=label_columns,
        drop_columns=drop_columns,
    )

    # Parsing kwargs:
    predict_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.PREDICT
    )

    # loading the model, and getting the model handler:
    model_handler = AutoMLRun.load_model(model_path=model, context=context)

    # Dropping label columns if necessary:
    if label_columns and all(label in dataset.columns for label in label_columns):
        dataset = dataset.drop(label_columns, axis=1)

    # Predicting:
    context.logger.info(f"making prediction by '{model_handler.model_name}'")
    y_pred = model_handler.model.predict(dataset, **predict_kwargs)

    if not label_columns:
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            label_columns = ["predicted_labels"]
        else:
            label_columns = [f"predicted_label_{i}" for i in range(y_pred.shape[1])]
    elif isinstance(label_columns, str):
        label_columns = [label_columns]

    pred_df = pd.concat([dataset, pd.DataFrame(y_pred, columns=label_columns)], axis=1)
    context.log_dataset("prediction", pred_df)

    # Drift Analysis:
    if (
        perform_drift_analysis is None
        and model_handler._model_artifact.feature_stats is not None
    ):
        perform_drift_analysis = True
    if perform_drift_analysis:
        context.logger.info("Performing drift analysis.")
        # Check if the model was logged with a sample set:
        if model_handler._model_artifact.feature_stats is None:
            raise ValueError(
                "Cannot perform drift analysis as the model artifact was not logged with a sample set."
            )
        # Produce the artifact:
        # TODO: Replace `sample_set_statistics` to `model_handler._model_artifact.spec.feature_stats` when MLRun's
        #       version is released.
        drift_analysis_artifacts = _perform_drift_analysis(
            context=context,
            sample_set_statistics=model_handler._model_artifact.feature_stats,
            inputs=pred_df,
            drift_threshold=drift_threshold,
            possible_drift_threshold=possible_drift_threshold,
        )
        for artifact in drift_analysis_artifacts:
            context.log_artifact(artifact)
