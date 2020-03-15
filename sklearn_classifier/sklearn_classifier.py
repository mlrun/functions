import os
import json
import importlib
from cloudpickle import dump

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, 
                             f1_score,
                             auc,
                             average_precision_score, 
                             roc_curve, 
                             confusion_matrix)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

from yellowbrick.classifier import ConfusionMatrix

from typing import Optional, Union, List

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import PlotArtifact, TableArtifact

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def _gcf_clear(plt):
    plt.cla()
    plt.clf()
    plt.close() 

def train_model(
    context: MLClientCtx,
    data_key: Union[DataItem, str],
    sample: int,
    label_column: str,
    model_key: str = "model", 
    test_size: float = 0.1,
    train_val_split: float = 0.75,
    rng: int = 1,
    score_method: str = 'micro',
    class_params: Union[DataItem, dict] = {},
    fit_params: Union[DataItem, dict] = {}
) -> None:
    """train a classifier.

    :param context:         the function context
    :param data_key:        ("raw") name of raw data file
    :param sample:          Selects the first n rows, or select a sample starting
                            from the first. If negative <-1, select a random sample
    :param label_column:    ground-truth (y) labels
    :param model_key:       ('model') name of model in artifact store, points to a directory
    :param test_size:       (0.1) test set size
    :param train_val_split: (0.75) Once the test set has been removed the 
                            training set gets this proportion.
    :param rng:             (1) sklearn rng seed
    :param score_method:     for multiclass classification
    :param class_params:    scikit-learn classifier params, either input as dict or DataItem from
                            a preceding Kubeflow pipeline step
    :param fit_params:      scikit-learn fit parameters, either input as dict or DataItem from
                            a preceding Kubeflow pipeline step
    """
    base_path = context.artifact_path
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path+"/models", exist_ok=True)
    os.makedirs(base_path+"/plots", exist_ok=True)
    
    srcfilepath = str(data_key)
    
    # get all data or a sample
    if (sample == -1) or (sample >= 1):
        # get all rows, or contiguous sample starting at row 1.
        raw = pq.read_table(srcfilepath).to_pandas().dropna()
        labels = raw.pop(label_column)
        raw = raw.iloc[:sample, :]
        labels = labels.iloc[:sample]
    else:
        # grab a random sample
        raw = pq.read_table(srcfilepath).to_pandas().dropna().sample(sample*-1)
        labels = raw.pop(label_column)

    # this used by plotting methods
    context.header = raw.columns.values
    
    # double split to generate 3 data sets: train, validation and test
    # with xtest,ytest set aside
    x, xtest, y, ytest = train_test_split(raw.values, labels.values, test_size=test_size, 
                                          random_state=rng)
   
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, 
                                                      train_size=train_val_split, 
                                                      random_state=rng)
    classes = list(range(xtrain.shape[1]))
    ytrainb = label_binarize(ytrain, classes=classes)
    yvalidb = label_binarize(yvalid, classes=classes)
    
    filepath = os.path.join(base_path, "test_set.csv")
    pd.concat([pd.DataFrame(data=xtest, columns= context.header), 
               pd.DataFrame(data=ytest, columns=[label_column])], axis=1).to_csv(filepath, index=False)
    context.log_artifact("test_set", local_path="test_set.csv")   

    class_params = json.loads(class_params.get())
    fit_params = json.loads(fit_params.get())

    # create ClassifierClass from string and instantiate
    classifier = class_params.pop("classifier", None)
    splits = classifier.split(".")
    clfclass = splits[-1]
    pkg_module = splits[:-1]
    ClassifierClass = getattr(importlib.import_module(".".join(pkg_module)), clfclass)
    
    if "callbacks" in fit_params:
        # create callbacks from strings and instantiate with parameter `evals_result`
        splits = fit_params.pop("callbacks", None).split(".")
        pkg_module = ".".join(splits[:-1])
        cb_fname = splits[-1]
        pkg_module = __import__(pkg_module, fromlist=[cb_fname])
        evals_result = dict()
        callbacks = [getattr(pkg_module, cb_fname)(evals_result)]
    
    model =  ClassifierClass(**class_params)

    model.fit(xtrain,
              ytrain,
#               eval_set=[(xtrain, ytrain), (xvalid, yvalid)],
#               eval_metric="auc",
#               callbacks=callbacks,
              **fit_params)
    
    dump(model, open(os.path.join(base_path, "models/model.pkl"), "wb"))
    
    context.log_artifact(model_key, local_path="models")
    
    # compute validation metrics
    ypred = model.predict(xvalid) # == best_label
    y_score = model.predict_proba(xvalid) # nobs x classes probs matrix
    
    average_precision = average_precision_score(yvalidb, y_score, average=score_method)

    context.log_result(f"accuracy", float(model.score(xvalid, yvalid)))
    context.log_result(f"rocauc", roc_auc_score(yvalidb, y_score))
    context.log_result(f"f1_score", f1_score(yvalid, ypred, average=score_method))
    context.log_result(f"avg_precscore", average_precision)
    
    # plot ROC using validation set
    plot_roc(context, 
             yvalidb, 
             y_score,
             key="roc")
    
    # plot training history metrics, save and log:

#     if "training" in evals_result:
#         train_auc = np.asarray(evals_result["training"]["auc"], dtype=np.float)
#         train_logloss = np.asarray(evals_result["training"]["binary_logloss"], dtype=np.float)
#         val_auc = np.asarray(evals_result["valid_1"]["auc"], dtype=np.float)
#         val_logloss = np.asarray(evals_result["valid_1"]["binary_logloss"], dtype=np.float)

#     plot_validation(
#         context, 
#         [(train_auc, val_auc), (train_logloss, val_logloss)], 
#         key="history", 
#         title="training metrics")

#     # plot confusion matrix using validation set
#     # TODO: replace with sklearn or yellowbrick classification report
    plot_confusion_matrix(
        context, 
        yvalid, 
        ypred,
        key="confusion",
        fmt="png")

#     plot feature importances
    if hasattr(model, "feature_importances_"):
        plot_importance(context, model, key=f"featimp")
    
def plot_validation(
    context: MLClientCtx,
    metrics, # list of tuple(vec, vec)
    key: str = "training-history",
    title: str = "auc and logloss",
    fmt="png"
):
    """Plot train and validation loss curves

    These curves represent the training round metrics from the training
    and validation sets.
    
    :param context:         the function context
    :param train_metric:    train metric
    :param valid_metric:    validation metric
    """
    _gcf_clear(plt)
    
    # generate plot
    for metric in metrics:
        plt.plot(metric[0])
        plt.plot(metric[1])

    plt.title(f"training history - {title}")
    plt.xlabel("epoch")
    plt.ylabel("")
    plt.legend(["auc:train", "auc:valid", "logloss:train", "logloss:valid"])

    # save figure and log artifact
    plt.savefig(os.path.join(context.artifact_path, f"plots/{key}.{fmt}"))
    context.log_artifact(PlotArtifact(key,  body=plt.gcf()), local_path=f"plots/{key}.{fmt}")
    
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
    fig.savefig(os.path.join(context.artifact_path, f"plots/{key}.{fmt}"))
    context.log_artifact(PlotArtifact(key,  body=fig), local_path=f"plots/{key}.{fmt}")

def plot_confusion_matrix(
    context: MLClientCtx, 
    labels, 
    predictions,
    key: str ="confusion_matrix",
    fmt: str = "png"
):
    """Create a confusion matrix.
    Plot and save a confusion matrix using test data from a
    modelline step.
    :param context:         function context
    :param labels:          test data labels
    :param predictions:     test data predictions
    """
    _gcf_clear(plt)

    cm = confusion_matrix(labels,
                          predictions,
                          sample_weight=None)
    #                     normalize="all")
    sns.heatmap(cm, annot=True, cmap="Blues")

    fig = plt.gcf()
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
    fig.savefig(os.path.join(context.artifact_path, f"plots/{key}.{fmt}"))
    context.log_artifact(PlotArtifact(f"{key}.{fmt}", body=fig), local_path=f"plots/{key}.{fmt}")

    # feature importances are also saved as a table:
    feature_imp.to_csv(os.path.join(context.artifact_path, key+".csv"))
    context.log_artifact(key+".csv", local_path=key+".csv")

    _gcf_clear(plt)