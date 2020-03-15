import os
import json
import importlib
from cloudpickle import load

import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize

from sklearn.metrics import (roc_auc_score, 
                             auc,
                             f1_score, 
                             average_precision_score, 
                             roc_curve, 
                             confusion_matrix)

from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from typing import Optional, Union, List, Tuple

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import TableArtifact, PlotArtifact

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def _gcf_clear(plt):
    plt.cla()
    plt.clf()
    plt.close()        

def test_classifier(
    context: MLClientCtx,
    models_dir: Union[DataItem, str], 
    test_set: DataItem,
    label_column: str,
    score_method: str = 'micro',
    key: str = ""
) -> None:
    """Test one or more classifier models against held-out dataset
    
    Using held-out test features, evaluates the peformance of the estimated model
    
    Can be part of a kubeflow pipeline as a test step that is run post EDA and 
    training/validation cycles
    
    :param context:         the function context
    :param models_dir:      artifact models representing a folder or a folder
    :param test_set:        test features and labels
    :param label_column:    column name for ground truth labels
    :param score_method:     for multiclass classification
    :param key:             key for results artifact
    """
    os.makedirs(os.path.join(context.artifact_path, "plots"), exist_ok=True)

    xtest = pd.read_csv(str(test_set))
    ytest = xtest.pop(label_column)

    context.header = list(xtest.columns.values)
    
    def _eval_model(model):
        # enclose all except model
        ytestb = label_binarize(ytest, classes=[0, 1, 2])
        clf = load(open(os.path.join(str(models_dir), model), "rb"))
        if callable(getattr(clf, "predict_proba")):
            y_score = clf.predict_proba(xtest.values)
            ypred = clf.predict(xtest.values)
            plot_roc(context, ytestb, y_score, key=f"roc")
        else:
            ypred = clf.predict(xtest.values) # refactor
            y_score = None
        plot_confusion_matrix(context, 
                                ytest, 
                                ypred, 
                                classes=context.header[:-1],
                                key=f"confusion")
        if hasattr(clf, "feature_importances_"):
            print(clf)
            plot_importance(context, clf, key=f"featimp")
        average_precision = average_precision_score(ytestb, y_score, average=score_method)
        context.log_result(f"accuracy", float(clf.score(xtest.values, ytest.values)))
        context.log_result(f"rocauc", roc_auc_score(ytestb, y_score))
        context.log_result(f"f1_score", f1_score(ytest.values, ypred, average=score_method))
        context.log_result(f"avg_precscore", average_precision)

    
    best_model = None
    for model in os.listdir(str(models_dir)):
        if model.endswith('.pkl'):
            _eval_model(model)
            # HACK: there is only one model here
            best_model = model

    # log 'best model' as artifact
    context.log_artifact('TODAYS-MODELS-TEST-REPORT', local_path=best_model)
    context.log_artifact('DEPLOY', body=b'true', local_path='DEPLOY')