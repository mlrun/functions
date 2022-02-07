from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.model_selection import train_test_split

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.frameworks.auto_mlrun import AutoMLRun
from mlrun import feature_store as fs
from mlrun.api.schemas import ObjectKind
from mlrun.utils.helpers import create_class, create_function


def _parse_kwargs(kwargs: Dict) -> Tuple[Dict, Dict, Dict]:
    train_kw, fit_kw, model_class_kw = {}, {}, {}

    for key, val in kwargs.items():
        if key.startswith("TRAIN_"):
            train_kw[key.replace("TRAIN_", "")] = val

        elif key.startswith("FIT_"):
            train_kw[key.replace("FIT_", "")] = val

        elif key.startswith("MODEL_CLASS_"):
            model_class_kw[key.replace("MODEL_CLASS_", "")] = val

    return train_kw, fit_kw, model_class_kw


def train(
    context: MLClientCtx,
    dataset: DataItem,
    drop_columns: List[str] = None,
    model_class: str = None,
    model_name: str = "model",
    tag: str = "",
    label_columns: Optional[Union[str, List[str]]] = None,
    sample_set: DataItem = None,
    test_set: DataItem = None,
    train_test_split_size: float = None,
    artifacts: Dict = None,
    kwargs: Dict[str, Any] = None,
):
    # Validate inputs:
    # Check if only one of them is supplied:
    if (test_set is None) == (train_test_split_size is None):
        raise TypeError(
            f"Provide only one of test_set model and train_test_split_size"
        )

    if dataset.meta and dataset.meta.kind == ObjectKind.feature_vector:
        # feature-vector case:
        dataset = fs.get_offline_features(dataset.meta.uri, drop_columns=drop_columns).to_dataframe()
        label_columns = label_columns or dataset.meta.status.label_column
    else:
        # simple URL case:
        dataset = dataset.as_df()
        if drop_columns:
            dataset = dataset.drop(drop_columns, axis=1)

    # Parsing kwargs:
    train_kw, fit_kw, model_class_kw = _parse_kwargs(kwargs)

    # Check if model or function:
    if hasattr(model_class, 'train'):
        # TODO: Need to call: model(), afterwards to start the train function.
        model = create_function(f'{model_class}.train')
    else:
        # Creating model instance:
        model = create_class(model_class)(**model_class_kw)

    x = dataset.drop(label_columns, axis=1)
    y = dataset[label_columns]
    if train_test_split_size:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=train_test_split_size
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
        test_set=test_set,
        X_test=x_test,
        y_test=y_test,
        artifacts=artifacts,
    )

    model = model.fit(x_train, y_train, **fit_kw)
    pass
