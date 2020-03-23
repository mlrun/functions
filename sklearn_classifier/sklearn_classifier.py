import json
import os
from importlib import import_module
from inspect import getfullargspec, FullArgSpec
from cloudpickle import dump, load
import itertools

import sklearn
from sklearn import metrics
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.testing import all_estimators
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn import metrics

from typing import Union, List, Any, Optional
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import PlotArtifact

skversion = sklearn.__version__

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def _gcf_clear(plt):
    """Utility to clear matplotlib figure

    Run this inside every plot method before calling any matplotlib
    methods

    :param plot:    matloblib figure object
    """
    plt.cla()
    plt.clf()
    plt.close()


def _create_class(pkg_class: str):
    """Create a class from a package.module.class string

    :param pkg_class:  full class location,
                       e.g. "sklearn.model_selection.GroupKFold"
    """
    splits = pkg_class.split(".")
    clfclass = splits[-1]
    pkg_module = splits[:-1]
    class_ = getattr(import_module(".".join(pkg_module)), clfclass)
    return class_

def _create_function(pkg_func: list):
    """Create a function from a package.module.function string

    :param pkg_func:  full function location,
                      e.g. "sklearn.feature_selection.f_classif"
    """
    splits = pkg_func.split(".")
    pkg_module = ".".join(splits[:-1])
    cb_fname = splits[-1]
    pkg_module = __import__(pkg_module, fromlist=[cb_fname])
    function_ = getattr(pkg_module, cb_fname)
    return function_

def get_model_configs(
    my_models: Union[str, List[str]],
    class_key = "CLASS",
    fit_key = "FIT",
    meta_key = "META",
) -> Union[dict, List[dict]]:
    """build sklearn model configuration parameters
    
    Take (full) class name of an scikit-learn model 
    and retrieve its `class` and `fit` parameters and
    their default values.
    
    Also returns some useful metadata values for the class
    """
    # get a list of all sklearn estimators
    estimators = all_estimators()
    def _get_estimator(pkg_class):
        """find a specific class in a list of sklearn estimators"""
        my_class = pkg_class.split(".")[-1]
        return list(filter(lambda x: x[0] == my_class, estimators))[0]

    # find estimators corresponding to my_models list
    my_estimators = []
    my_models = [my_models] if isinstance(my_models, str) else my_models
    for model in my_models:
        estimator_name, estimator_class = _get_estimator(model)
        my_estimators.append((estimator_name, estimator_class))

    # get class and fit specs
    estimator_specs = []
    for an_estimator in my_estimators:
        estimator_specs.append((an_estimator[0], # model only name
                                getfullargspec(an_estimator[1]), # class params
                                getfullargspec(an_estimator[1].fit), # fit params
                                an_estimator[1])) # package.module.model

    model_configs = []

    for estimator in estimator_specs:
        model_json = {class_key: {}, fit_key: {}}
        fit_params = {}

        for i, key in enumerate(model_json.keys()):
            f = estimator[i+1]
            args_paired = []
            defs_paired = []

            # reverse the args since there are fewer defaults than args
            args = f.args
            args.reverse()
            n_args = len(args)

            defs = f.defaults
            if defs is None:
                defs = [defs]
            defs = list(defs)
            defs.reverse()
            n_defs = len(defs)

            n_smallest = min(n_args, n_defs)
            n_largest = max(n_args, n_defs)

            # build 2 lists that can be concatenated
            for ix in range(n_smallest):
                if args[ix] is not "self":
                    args_paired.append(args[ix])
                    defs_paired.append(defs[ix])

            for ix in range(n_smallest, n_largest):
                if ix is not 0 and args[ix] is not "self":
                    args_paired.append(args[ix])
                    defs_paired.append(None)
               # concatenate lists into appropriate structure
            model_json[key] = dict(zip(reversed(args_paired), reversed(defs_paired)))

        model_json[meta_key] = {}
        model_json[meta_key]["sklearn_version"] = skversion
        model_json[meta_key]["class"] = ".".join([estimator[3].__module__, estimator[0]])
        model_configs.append(model_json)
    if len(model_configs) == 1:
        # do we want to log this modified model as an artifact?
        return model_configs[0]
    else:
        # do we want to log this modified model as an artifact?
        return model_configs

def update_model_config(
    config: dict,
    new_class: dict,
    new_fit: dict,
    class_key: str = "CLASS",
    fit_key: str = "FIT"
):
    """Update model config json
    
    Not used until we refactor as per the TODO
        
    This function is essential since there are modifications in class
    and fit params that must be made (callbacks are a good example, without
    which there is no training history available)
    
    TODO:  currently a model config contains 2 keys, but this will likely
    expand to include other functions beyond class and fit. So need to expand 
    this to a list of Tuple(str, dict), where `str` corresponds to a key
    in the model config and `dict` contains the params and their new values.
    
    :param config:      original model definition containing 2 keys, CLASS and FIT
    :param new_class:   new class key-values
    :param new_fit:     new fit key-values
    """
    config[class_key].update(new_class)
    config[fit_key].update(new_fit)
    
    return config

def train_model(
    context: MLClientCtx,
    model_pkg_class: str,
    data_key: Union[DataItem, str],
    sample: int,
    label_column: str,
    model_key: str = "model",
    test_size: float = 0.05,
    train_val_split: float = 0.75,
    test_set_key: str = "test_set",
    rng: int = 1,
    models_dir: str = "models",
    plots_dir: str = "plots",
    score_method: str = "micro",
    class_params_updates: Union[DataItem, dict] = {},
    fit_params_updates: Union[DataItem, dict] = {},
) -> None:
    """train a classifier.

    :param context:           the function context
    :param model_pkg_class:   the model to train, e.g, "sklearn.neural_networks.MLPClassifier", 
                              or json model config
    :param data_key:          ("raw") name of raw data file
    :param sample:            Selects the first n rows, or select a sample
                              starting from the first. If negative <-1, select
                              a random sample
    :param label_column:      ground-truth (y) labels
    :param model_key:         ("model") name of model in artifact store,
                              points to a directory
    :param test_size:         (0.05) test set size
    :param train_val_split:   (0.75) Once the test set has been removed the
                              training set gets this proportion.
    :param test_set_key:      store the test data set under this key in the
                              artifact store
    :param rng:               (1) sklearn rng seed
    :param models_dir:        models subfolder on artifact path
    :param plots_dir:         plot subfolder on artifact path
    :param score_method:      for multiclass classification
    :param class_updates:     update these scikit-learn classifier params,
                              input as a dict
    :param fit_updates:       update scikit-learn fit parameters, input as
                              a dict.
    """
    base_path = context.artifact_path
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, plots_dir), exist_ok=True)
    os.makedirs(os.path.join(base_path, models_dir), exist_ok=True)
    
    # extract file name from DataItem
    srcfilepath = str(data_key)
    
    # TODO: this should be part of data"s metadata dealt with in another step get a data set, sample, etc...
    # get all data or a sample
    if (sample == -1) or (sample >= 1):
        # get all rows, or contiguous sample starting at row 1.
        raw = pq.read_table(srcfilepath).to_pandas().dropna()
        labels = raw.pop(label_column)
        raw = raw.iloc[:sample, :]
        labels = labels.iloc[:sample]
    else:
        # grab a random sample
        raw = pq.read_table(srcfilepath).to_pandas().dropna().sample(sample * -1)
        labels = raw.pop(label_column)

    # TODO: this should be part of data"s metadata dealt with in another step
    context.header = raw.columns.values
    
    # TODO: all of this should be part of a spitter component that does cv too, dealt with in another step
    # make a hot encode copy of labels before the split
    yb = label_binarize(labels, classes=labels.unique())
    # double split to generate 3 data sets: train, validation and test
    # with xtest,ytest set aside
    x, xtest, y, ytest = train_test_split(np.concatenate([raw, yb], axis=1), labels, test_size=test_size, random_state=rng)
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, train_size=train_val_split, random_state=rng)
    # extract the hot_encoded labels
    ytrainb = xtrain[:, -yb.shape[1]:].copy()
    xtrain = xtrain[:, :-yb.shape[1]].copy()
    # extract the hot_encoded labels
    yvalidb = xvalid[:, -yb.shape[1]:].copy()
    xvalid = xvalid[:, :-yb.shape[1]].copy()
    # extract the hot_encoded labels
    ytestb = xtest[:, -yb.shape[1]:].copy()
    xtest = xtest[:, :-yb.shape[1]].copy()                                      
    # set-aside test_set
    test_set = pd.concat(
        [pd.DataFrame(data=xtest, columns=context.header),
         pd.DataFrame(data=ytest.values, columns=[label_column])],
        axis=1,)
    filepath = os.path.join(base_path, test_set_key + ".pqt")
    test_set.to_parquet(filepath, index=False)
    context.log_artifact(test_set_key, local_path=test_set_key + ".pqt")

    if model_pkg_class.endswith(".json"):
        model_config = json.load(open(model_pkg_class, "r"))
    else:
        # load the model config
        model_config = get_model_configs(model_pkg_class)

    # get update params if any
    if isinstance(class_params_updates, DataItem):
        class_params_updates = json.loads(class_params_updates.get())
    if isinstance(fit_params_updates, DataItem):
        fit_params_updates = json.loads(fit_params_updates.get())
    # update the parameters            
    # add data to fit params
    fit_params_updates.update({"X": xtrain,"y": ytrain.values})
    
    model_config["CLASS"].update(class_params_updates)
    model_config["FIT"].update(fit_params_updates)
    
    # create class and fit
    ClassifierClass = _create_class(model_config["META"]["class"])
    model = ClassifierClass(**model_config["CLASS"])
    model.fit(**model_config["FIT"])

    # save model
    filepath = os.path.join(base_path, f"{models_dir}/{model_key}.pkl")
    try:
        dump(model, open(filepath, "wb"))
        context.log_artifact(model_key, local_path=models_dir)
    except Exception as e:
        print("SERIALIZE MODEL ERROR:", str(e))

    # compute validation metrics
    ypred = model.predict(xvalid)
    y_score = model.predict_proba(xvalid)
    context.logger.info(f"y_score.shape {y_score.shape}")
    context.logger.info(f"yvalidb.shape {yvalidb.shape}")
    if yvalidb.shape[1] > 1:
        # label encoding was applied:
        average_precision = metrics.average_precision_score(yvalidb[:,:-1],
                                                            y_score,
                                                            average=score_method)
        context.log_result(f"rocauc", metrics.roc_auc_score(yvalidb, y_score))
    else:
        average_precision = metrics.average_precision_score(yvalidb,
                                                            y_score[:, 1],
                                                            average=score_method)
        context.log_result(f"rocauc", metrics.roc_auc_score(yvalidb, y_score[:, 1]))
        
    context.log_result(f"avg_precscore", average_precision)
    context.log_result(f"accuracy", float(model.score(xvalid, yvalid)))
    context.log_result(f"f1_score", metrics.f1_score(yvalid, ypred,
                                             average=score_method))
    

    # validation plots
    
    plot_roc(context, yvalidb, y_score)
    plot_confusion_matrix(context, yvalid, ypred, key="confusion", fmt="png")

def plot_roc(
    context,
    y_labels,
    y_probs,
    key="roc",
    plots_dir: str = "plots",
    fmt="png",
    x_label: str = "false positive rate",
    y_label: str =  "true positive rate",
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
    :param x_label:      ("false positive rate") x-axis labels
    :param y_label:      ("true positive rate") y-axis labels
    :param title:        ("roc curve") title of plot
    :param legend_loc:   ("best") location of plot legend
    """
    # clear matplotlib current figure
    _gcf_clear(plt)
    
    # draw 45 degree line
    plt.plot([0, 1], [0, 1], "k--")
    
    # labelling
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
