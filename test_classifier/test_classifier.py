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
    
def plot_roc(
    context: MLClientCtx, 
    y_labels,
    y_probs,
    key="roc",
    fmt="png"
):
    """Plot an ROC curve from test data saved in an artifact store.
    
    :param context:         function context
    :param y_labels:        test data labels
    :param y_probs:         test data 
    """
    _gcf_clear(plt)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend(loc="best")
    for i in range(y_labels.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_labels[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"class {i}")

    fig = plt.gcf()
    os.makedirs(os.path.join(context.artifact_path, "plots"), exist_ok=True)
    fig.savefig(os.path.join(context.artifact_path, f"plots/{key}.{fmt}"))
    context.log_artifact(PlotArtifact(key,  body=fig), local_path=f"plots/{key}.{fmt}")

def plot_confusion_matrix(
    context: MLClientCtx, 
    labels, 
    predictions,
    classes,
    key: str ="confusion",
    fmt: str = "png"
):
    """Create a confusion matrix.
    Plot and save a confusion matrix using test data from a
    pipeline step.
    :param context:         function context
    :param labels:          test data labels
    :param predictions:     test data predictions
    """
    _gcf_clear(plt)

    cm = _plot_confusion_matrix(labels, predictions, classes=classes, 
                                title=key, normalize=True)

    fig = plt.gcf()
    os.makedirs(os.path.join(context.artifact_path, "plots"), exist_ok=True)
    fig.savefig(os.path.join(context.artifact_path, f"plots/{key}.{fmt}"))
    context.log_artifact(PlotArtifact(key,  body=fig), local_path=f"plots/{key}.{fmt}")

    _gcf_clear(plt)

def plot_importance(
    context,
    model,
    key: str = "feature-importances",
    fmt = "png"
):
    """Display estimated feature importances.
    :param context:     function context
    :param model:       fitted lightgbm model
    """
    _gcf_clear(plt)
    
    # create a feature importance table with desired labels
    zipped = zip(model.feature_importances_, context.header)

    feature_imp = pd.DataFrame(sorted(zipped), columns=["freq","feature"]
                                ).sort_values(by="freq", ascending=False)

    plt.figure(figsize=(20, 10))
    sns.barplot(x="freq", y="feature", data=feature_imp)
    plt.title("features")
    plt.tight_layout()

    fig = plt.gcf()
    os.makedirs(os.path.join(context.artifact_path, "plots"), exist_ok=True)
    fig.savefig(os.path.join(context.artifact_path, f"plots/{key}.{fmt}"))
    context.log_artifact(PlotArtifact(f"{key}.{fmt}", body=fig), local_path=f"plots/{key}.{fmt}")

    # feature importances are also saved as a table:
    feature_imp.to_csv(os.path.join(context.artifact_path, key+".csv"))
    context.log_artifact(key+".csv", local_path=key+".csv")

    _gcf_clear(plt)

def _plot_confusion_matrix(y_true, y_pred, classes=["neg", "pos"], 
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """This can be deprecated once intel python upgrades scikit-learn to >0.22
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    https://scikit-learn.org/0.21/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax    