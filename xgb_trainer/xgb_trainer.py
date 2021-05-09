
from mlrun.mlutils.data import get_sample, get_splits
from mlrun.mlutils.models import gen_sklearn_model, eval_model_v2
from mlrun.utils.helpers import create_class

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import PlotArtifact, TableArtifact

from cloudpickle import dumps
import pandas as pd
import os
from typing import List, Union


def _gen_xgb_model(model_type: str, xgb_params: dict):
    """generate an xgboost model

    Multiple model types that can be estimated using
    the XGBoost Scikit-Learn API.

    Input can either be a predefined json model configuration or one
    of the five xgboost model types: "classifier", "regressor", "ranker",
    "rf_classifier", or "rf_regressor".

    In either case one can pass in a params dict to modify defaults values.

    Based on `mlutils.models.gen_sklearn_model`, see the function
    `sklearn_classifier` in this repository.

    :param model_type: one of "classifier", "regressor",
                       "ranker", "rf_classifier", or
                      "rf_regressor"
    :param xgb_params: class init parameters
    """
    # generate model and fit function
    mtypes = {
        "classifier": "xgboost.XGBClassifier",
        "regressor": "xgboost.XGBRegressor",
        "ranker": "xgboost.XGBRanker",
        "rf_classifier": "xgboost.XGBRFClassifier",
        "rf_regressor": "xgboost.XGBRFRegressor"
    }
    if model_type.endswith("json"):
        model_config = model_type
    elif model_type in mtypes.keys():
        model_config = mtypes[model_type]
    else:
        raise Exception("unrecognized model type, see help documentation")

    return gen_sklearn_model(model_config, xgb_params)


def train_model(
        context: MLClientCtx,
        model_type: str,
        dataset: Union[DataItem, pd.core.frame.DataFrame],
        label_column: str = "labels",
        encode_cols: dict = {},
        sample: int = -1,
        imbal_vec=[],
        test_size: float = 0.25,
        valid_size: float = 0.75,
        random_state: int = 1,
        models_dest: str = "models",
        plots_dest: str = "plots",
        eval_metrics: list = ["error", "auc"],
        file_ext: str = "parquet",
        test_set: str = "test_set"
) -> None:
    """train an xgboost model.

    Note on imabalanced data:  the `imbal_vec` parameter represents the measured
    class representations in the sample and can be used as a first step in tuning
    an XGBoost model.  This isn't a hyperparamter, merely an estimate that should
    be set as 'constant' throughout tuning process.

    :param context:           the function context
    :param model_type:        the model type to train, "classifier", "regressor"...
    :param dataset:           ("data") name of raw data file
    :param label_column:      ground-truth (y) labels
    :param encode_cols:       dictionary of names and prefixes for columns that are
                              to hot be encoded.
    :param sample:            Selects the first n rows, or select a sample
                              starting from the first. If negative <-1, select
                              a random sample
    :param imbal_vec:         ([]) vector of class weights seen in sample
    :param test_size:         (0.05) test set size
    :param valid_size:        (0.75) Once the test set has been removed the
                              training set gets this proportion.
    :param random_state:      (1) sklearn rng seed
    :param models_dest:       destination subfolder for model artifacts
    :param plots_dest:        destination subfolder for plot artifacts
    :param eval_metrics:      (["error", "auc"]) learning curve metrics
    :param file_ext:          format for test_set_key hold out data
    :param test-set:          (test_set) key of held out data in artifact store
    """
    # deprecate:
    models_dest = models_dest or "models"
    plots_dest = plots_dest or f"plots/{context.name}"

    # get a sample from the raw data
    raw, labels, header = get_sample(dataset, sample, label_column)

    # hot-encode
    if encode_cols:
        raw = pd.get_dummies(raw,
                             columns=list(encode_cols.keys()),
                             prefix=list(encode_cols.values()),
                             drop_first=True)

    # split the sample into train validate, test and calibration sets:
    (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = \
        get_splits(raw, labels, 3, test_size, valid_size, random_state)

    # save test data
    context.log_dataset(test_set, df=pd.concat([xtest, ytest], axis=1), format=file_ext, index=False)

    # get model config
    model_config = _gen_xgb_model(model_type, context.parameters.items())

    # create model instance
    XGBBoostClass = create_class(model_config["META"]["class"])
    model = XGBBoostClass(**model_config["CLASS"])

    # update the model config with training data and callbacks
    model_config["FIT"].update({"X": xtrain,
                                "y": ytrain.values,
                                "eval_set": [(xtrain, ytrain), (xvalid, yvalid)],
                                "eval_metric": eval_metrics})

    # run the fit
    model.fit(**model_config["FIT"])

    # evaluate model
    eval_metrics = eval_model_v2(context, xvalid, yvalid, model)

    model_bin = dumps(model)
    context.log_model("model", body=model_bin,
                      artifact_path=os.path.join(context.artifact_path, models_dest),
                      # model_dir=models_dest,
                      model_file="model.pkl")