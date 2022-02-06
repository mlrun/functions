import mlrun
import warnings

from typing import List, Dict, Union
from sklearn.model_selection import train_test_split
from importlib import import_module
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.frameworks.auto_mlrun import AutoMLRun
from inspect import _empty, signature

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def get_class_fit(package_module_class: str) -> Dict[str, Union[Dict[str, ], str]]:
    """generate a model config
    :param package_module_class:  str description of model, e.g.
        `sklearn.ensemble.RandomForestClassifier`
    """
    package, module, klass = package_module_class.split(".")

    # getting the model:
    model = getattr(import_module(f"{package}.{module}"), klass)

    # model.fit() signature:
    fit_signature = dict(signature(model().fit).parameters)

    # TODO: How do we know the difference between default = None, and no default?
    for parameter, val in fit_signature.items():
        fit_signature[parameter] = None if val.default is _empty else val.default

    return {
        "CLASS": model().get_params(),
        "FIT": fit_signature,
        "META": {
            "package_version": import_module(package).__version__,
            "class": package_module_class,
        },
    }


def create_class(pkg_class: str):
    """Create a class from a package.module.class string
    :param pkg_class:  full class location,
                       e.g. "sklearn.model_selection.GroupKFold"
    """
    package, module, klass = pkg_class.split(".")
    class_ = getattr(import_module(f"{package}.{module}"), klass)
    return class_


def _gen_model_config(model_class, model_params):
    model_config = get_class_fit(model_class)

    for param in model_params:
        if param.startswith("CLASS_"):
            model_config["CLASS"][param[6:]] = model_params[param]

        if param.startswith("FIT_"):
            model_config["FIT"][param[4:]] = model_params[param]
    return model_config


def train(context: MLClientCtx,
          dataset: DataItem,
          model_class: str,
          label_column: str = 'label',
          model_name: str = 'trained_model',
          test_size: float = 0.2,
          artifacts: List[str] = []):
    """
    """

    # Set model config file
    model_config = _gen_model_config(model_class, dict(context.parameters.items()))

    # Pull DataFrame from DataItem
    dataset = dataset.as_df() if type(dataset) == mlrun.datastore.base.DataItem else dataset

    # Split according to test_size
    X = dataset[dataset.columns[dataset.columns != label_column]]
    y = dataset[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Update config with the new split
    model_config["FIT"].update({"X": X_train, "y": y_train})

    # ?
    model_class = create_class(model_config["META"]["class"])
    model = model_class(**model_config["CLASS"])

    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    AutoMLRun.apply_mlrun(
        model_name=model_name, model=model, context=context, x_validation=X_test, y_validation=y_test
    )

    # Train our model
    model.fit(model_config["FIT"]["X"], model_config["FIT"]["y"])


def evaluate(context: MLClientCtx,
             dataset: DataItem,
             model_path: str,
             artifacts: List[str],
             label_column: str = 'label'):

    # Pull DataFrame from DataItem
    dataset = dataset.as_df() if type(dataset) == mlrun.datastore.base.DataItem else dataset

    model_handler = AutoMLRun.load_model(model_path)

    X_test = dataset[dataset.columns[dataset.columns != label_column]]
    y_test = dataset[label_column]

    AutoMLRun.apply_mlrun(model_handler.model, y_test=y_test, model_path=model_path)

    model_handler.model.predict(X_test)


def predict(context: MLClientCtx,
            dataset: DataItem,
            model_path: str,
            artifacts: List[str],
            label_column: str = None):

    # Pull DataFrame from DataItem
    dataset = dataset.as_df() if type(dataset) == mlrun.datastore.base.DataItem else dataset

    if label_column:
        dataset = dataset[dataset.columns[dataset.columns != label_column]]

    model_handler = AutoMLRun.load_model(model_path)
    model_handler.model.predict(dataset)