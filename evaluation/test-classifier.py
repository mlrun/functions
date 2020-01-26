import os
import importlib
from cloudpickle import load

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from typing import Optional, Union, List

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import TableArtifact, PlotArtifact


def test_model(
    context: Optional[MLClientCtx],
    model: Union[DataItem, str],
    xtest, 
    ytest,
    target_path: str = '',
    name: str = '',
    key: str = '',
    random_state = 1
) -> None:
    """Test a classifier model
    
    Using held-out test features, calls `model.predict(xtest)` and evaluates the accuracy of the 
    estimated model.
    
    Can be part of a kubeflow pipeline as a test step or called
    
    :param context:         the function context
    :param model:           estimated model file name as artifact store item
                            or pickle file name
    :param xtest:           test features file name as artifact store item
                            or pickle file name
    :param header:          (Optional) use if xtest does not have a header
    :param ytest:           test labels file name as artifact store 
                            item or pickle file name
    :param target_path:     folder location of files
    :param name:            destination name for test results
    :param key:             key for model artifact
    """
    # load model and data
    if isinstance(model, DataItem):
        clf = load(open(str(model), 'rb'))
    else:
        clf = load(open(model, 'rb'))

    if isinstance(xtest, DataItem):
        xtest = pd.read_parquet(str(xtest))
        ytest = pd.read_parquet(str(ytest))
    else:
        xtest = pd.read_parquet(xtest)
        ytest = pd.read_parquet(ytest)
    
    if callable(getattr(clf, 'predict_proba')):
        ypred_probs = clf.predict_proba(xtest)[:, 1]
        ypred = np.where(ypred_probs >= 0.5, 1, 0)
        plot_roc(context, ytest, ypred_probs, target_path)
    else:
        ypred = clf.predict(xtest)
        ypred_probs = None
    
    plot_confusion_matrix(context, ytest, ypred, target_path)

    if hasattr(clf, 'feature_importances_'):
        plot_importance(context, clf, xtest.columns.values, target_path)

def _gcf_clear(plt):
    plt.cla()
    plt.clf()
    plt.close()        

def plot_roc(
    context: MLClientCtx, 
    y_labels,
    y_probs,
    target_path: str = '',
    name='roc.png',
    key='roc',
    fmt='png'
):
    """Plot an ROC curve from test data saved in an artifact store.
    
    :param context:         function context
    :param y_labels:        test data labels
    :param y_probs:         test data 
    """
    fpr_xg, tpr_xg, _ = roc_curve(y_labels, y_probs)
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr_xg, tpr_xg, label="roc")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend(loc="best")
    fig = plt.gcf()

    plotpath = os.path.join(target_path, name)
    fig.savefig(plotpath, format=fmt)
    context.log_artifact(PlotArtifact(key, body=fig))

    _gcf_clear(plt)

def plot_confusion_matrix(
    context: MLClientCtx, 
    labels, 
    predictions,
    target_path: str = '', 
    name: str ="confusion.png", 
    key: str ='confusion_matrix',
    fmt: str = 'png'
):
    """Create a confusion matrix.
    Plot and save a confusion matrix using test data from a
    pipeline step.

    :param context:         function context
    :param labels:          test data labels
    :param predictions:     test data predictions
    """
    cm = confusion_matrix(labels,
                            predictions,
                            sample_weight=None,
                            normalize='all')
    sns.heatmap(cm, annot=True, cmap="Blues")
    plotpath = os.path.join(target_path, name)
    fig = plt.gcf()
    fig.savefig(plotpath, format=fmt)
    context.log_artifact(PlotArtifact(key, body=fig))

    _gcf_clear(plt)

def plot_importance(
    context,
    model,
    header: List = [],
    target_path: str = '',
    name: str = 'feature-importances.png',
    key: str = 'feature-importances',
    fmt = 'png'
):
    """Display estimated feature importances.

    :param context:     function context
    :param model:       fitted lightgbm model
    :param header:      list of feature names
    """
    # create a feature importance table with desired labels
    zipped = zip(model.feature_importances_, header)

    feature_imp = pd.DataFrame(sorted(zipped), columns=['freq','feature']
                                ).sort_values(by="freq", ascending=False)

    plt.figure(figsize=(20, 10))
    sns.barplot(x="freq", y="feature", data=feature_imp)
    plt.title('LightGBM Features')
    plt.tight_layout()
    fig = plt.gcf()
    plotpath = os.path.join(target_path, name)
    fig.savefig(plotpath, format='png')
    context.log_artifact(PlotArtifact(key + '-plot', body=fig))

    # feature importances are also saved as a table:
    tablepath = os.path.join(target_path, key + '-table.csv')
    feature_imp.to_csv(tablepath)
    context.log_artifact(TableArtifact(key + '-table', target_path=tablepath))

    # to ensure we don't overwrite this figure when creating the next:
    _gcf_clear(plt)
