import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import pandas as pd
import mlrun

from mlrun.datastore import DataItem
from mlrun.artifacts import get_model, update_model
from mlrun.mlutils.models import eval_model_v2
from cloudpickle import load
from urllib.request import urlopen


def test_classifier(
    context,
    models_path: DataItem,
    test_set: DataItem,
    label_column: str,
    score_method: str = "micro",
    plots_dest: str = "",
    model_evaluator=None,
    default_model: str = "model.pkl",
    predictions_column: str = "yscore",
    model_update=True,
) -> None:
    """Test one or more classifier models against held-out dataset

    Using held-out test features, evaluates the peformance of the estimated model

    Can be part of a kubeflow pipeline as a test step that is run post EDA and
    training/validation cycles

    :param context:            the function context
    :param models_path:        artifact models representing a file or a folder
    :param test_set:           test features and labels
    :param label_column:       column name for ground truth labels
    :param score_method:       for multiclass classification
    :param plots_dest:         dir for test plots
    :param model_evaluator:    NOT IMPLEMENTED: specific method to generate eval, passed in as string
                               or available in this folder
    :param predictions_column: column name for the predictions column on the resulted artifact
    :param model_update:       (True) update model, when running as stand alone no need in update
    """
    xtest = test_set.as_df()
    ytest = xtest.pop(label_column)

    try:
        model_file, model_obj, _ = get_model(models_path, suffix=".pkl")
        model_obj = load(open(model_file, "rb"))
    except Exception as a:
        raise Exception("model location likely specified")

    extra_data = eval_model_v2(context, xtest, ytest.values, model_obj)
    if model_obj and model_update == True:
        update_model(
            models_path,
            extra_data=extra_data,
            metrics=context.results,
            key_prefix="validation-",
        )

    y_hat = model_obj.predict(xtest)
    if y_hat.ndim == 1 or y_hat.shape[1] == 1:
        score_names = [predictions_column]
    else:
        score_names = [f"{predictions_column}_" + str(x) for x in range(y_hat.shape[1])]

    df = pd.concat([xtest, ytest, pd.DataFrame(y_hat, columns=score_names)], axis=1)
    context.log_dataset("test_set_preds", df=df, format="parquet", index=False)
