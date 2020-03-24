import os
import json
import importlib
from cloudpickle import load

import numpy as np
import pandas as pd

import sklearn
from sklearn import metrics
from sklearn.preprocessing import label_binarize
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
    key: str = "",
    plots_dir: str = "plots"
) -> None:
    """Test one or more classifier models against held-out dataset
    
    Using held-out test features, evaluates the peformance of the estimated model
    
    Can be part of a kubeflow pipeline as a test step that is run post EDA and 
    training/validation cycles
    
    :param context:         the function context
    :param models_dir:      artifact models representing a folder or a folder
    :param test_set:        test features and labels
    :param label_column:    column name for ground truth labels
    :param score_method:    for multiclass classification
    :param key:             key for results artifact (maybe just a dir of artifacts for test like plots_dir)
    :param plots_dir:       dir for test plots
    """
    os.makedirs(os.path.join(context.artifact_path, plots_dir), exist_ok=True)
    
    xtest = pd.read_parquet(str(test_set))
    ytest = xtest.pop(label_column)
    
    context.header = list(xtest.columns.values)
    
    def _eval_model(model):
        # enclose all except model
        ytestb = label_binarize(ytest, classes=list(range(xtest.shape[1])))
        clf = load(open(os.path.join(str(models_dir), model), "rb"))
        if callable(getattr(clf, "predict_proba")):
            y_score = clf.predict_proba(xtest.values)
            ypred = clf.predict(xtest.values)
            context.logger.info(f"y_score.shape {y_score.shape}")
            context.logger.info(f"ytestb.shape {ytestb.shape}")
            plot_roc(context, ytestb, y_score, key=f"roc", plots_dir=plots_dir)
        else:
            ypred = clf.predict(xtest.values) # refactor
            y_score = None
        plot_confusion_matrix(context, ytest, ypred, key="confusion", fmt="png")
        if hasattr(clf, "feature_importances_"):
            plot_importance(context, clf, key=f"featimp")
        average_precision = metrics.average_precision_score(ytestb[:,:-1], y_score, average=score_method)
        context.log_result(f"accuracy", float(clf.score(xtest.values, ytest.values)))
        context.log_result(f"rocauc", metrics.roc_auc_score(ytestb, y_score))
        context.log_result(f"f1_score", metrics.f1_score(ytest.values, ypred, average=score_method))
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
    context,
    y_labels,
    y_probs,
    key="roc",
    plots_dir: str = "plots",
    fmt="png",
    fpr_label: str = "false positive rate",
    tpr_label: str =  "true positive rate",
    title: str = "roc curve",
    legend_loc: str = "best"
):
    """plot roc curves
    
    TODO:  add averaging method (as string) that was used to create probs, 
    display in legend
    
    :param context:      the function context
    :param y_labels:     ground truth labels, hot encoded for multiclass  
    :param y_probs:      model prediction probabilities
    :param key:          ("roc") key of plot in artifact store
    :param plots_dir:    ("plots") destination folder relative path to artifact path
    :param fmt:          ("png") plot format
    :param fpr_label:    ("false positive rate") x-axis labels
    :param tpr_label:    ("true positive rate") y-axis labels
    :param title:        ("roc curve") title of plot
    :param legend_loc:   ("best") location of plot legend
    """
       # clear matplotlib current figure
    _gcf_clear(plt)
    
    # draw 45 degree line
    plt.plot([0, 1], [0, 1], "k--")
    
    # labelling
    plt.xlabel(fpr_label)
    plt.ylabel(tpr_label)
    plt.title(title)
    plt.legend(loc=legend_loc)
    
    # single ROC or mutliple
    if y_labels.shape[1] > 1:
        # data accummulators by class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_labels[:,:-1].shape[1]):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_labels[:, i], y_probs[:, i], pos_label=1)
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"class {i}")
    else:
        fpr, tpr, _ = metrics.roc_curve(y_labels, y_probs[:, 1], pos_label=1)
        plt.plot(fpr, tpr, label=f"positive class")
        
    fname = f"{plots_dir}/{key}.{fmt}"
    plt.savefig(os.path.join(context.artifact_path, fname))
    context.log_artifact(PlotArtifact(key, body=plt.gcf()), local_path=fname)
    

def plot_confusion_matrix(
    context: MLClientCtx,
    labels,
    predictions,
    key: str = "confusion_matrix",
    plots_dir: str = "plots",
    colormap: str = "Blues",
    fmt: str = "png",
    sample_weight=None
):
    """Create a confusion matrix.
    Plot and save a confusion matrix using test data from a
    modelline step.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    
    TODO: fix label alignment
    TODO: consider using another packaged version
    TODO: refactor to take params dict for plot options

    :param context:         function context
    :param labels:          validation data ground-truth labels
    :param predictions:     validation data predictions
    :param key:             str
    :param plots_dir:       relative path of plots in artifact store
    :param colormap:        colourmap for confusion matrix
    :param fmt:             plot format
    :param sample_weight:   sample weights
    """
    _gcf_clear(plt)
    
    cm = metrics.confusion_matrix(labels, predictions, sample_weight=None)
    sns.heatmap(cm, annot=True, cmap=colormap, square=True)

    fig = plt.gcf()
    fname = f"{plots_dir}/{key}.{fmt}"
    fig.savefig(os.path.join(context.artifact_path, fname))
    context.log_artifact(PlotArtifact(key, body=fig), local_path=fname)

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
