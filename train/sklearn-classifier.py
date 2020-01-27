import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from typing import Optional, Union
import os
import importlib
from cloudpickle import dump

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import TableArtifact, PlotArtifact

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train(
    context: Optional[MLClientCtx] = None,
    SKClassifier: str  = '',
    callbacks  = [],
    xtrain: Union[DataItem, str] = '',
    ytrain: Union[DataItem, str] = '',
    xvalid: Union[DataItem, str] = '',
    yvalid: Union[DataItem, str] = '',
    target_path: str = '',
    name: str = '',
    key: str = '',
    verbose: bool = False,
    random_state = 1
) -> None:
    """Train and save an Scikitlearn model.
    
    The data source can either be a string file name or an artifact item.
    
    The header is eith a list of column names, an artifact header item, or None.
    
    
    :param context:         the function context
    :param SKClassifier:    string module and classname of classifier
    :param callbacks:       sklearn classifier fit function callbacks
    :param xtrain:          
    :param ytrain:
    :param xvalid:
    :param yvalid:
    :param target_path:     folder location of files
    :param name:            destination name for model file
    :param key:             key for model artifact
    :param verbose :        (False) show metrics for training/validation steps.
    :param random_state:    (1) sklearn rng seed
    
    example callbacks:
    ```
    from lightgbm import record_evaluation
    eval_results = dict()
    callbacks = [record_evaluation(eval_results)]
    ```
    """
    # load data
    xtrain = pd.read_parquet(str(xtrain), engine='pyarrow')
    ytrain = pd.read_parquet(str(ytrain), engine='pyarrow')
    xvalid = pd.read_parquet(str(xvalid), engine='pyarrow')
    yvalid = pd.read_parquet(str(yvalid), engine='pyarrow')

    # create classifier class from string and instantiate
    splits = SKClassifier.split(".")
    clfclass = getattr(importlib.import_module(".".join(splits[:-1])), splits[-1])
    model = clfclass(random_state=random_state, verbose=int(verbose == True))

    model.fit(xtrain, 
              ytrain,
              eval_set=[(xvalid, yvalid), (xtrain, ytrain)],
              eval_names=['valid', 'train'],
              callbacks=callbacks,
              verbose=verbose)
     
    context.log_result("train_accuracy", float(model.score(xtrain, ytrain)))
    
    # plot train and validation history, save and log
    loss = np.asarray(model.evals_result_['train']['binary_logloss'], dtype=np.float)
    val_loss = np.asarray(model.evals_result_['valid']['binary_logloss'], dtype=np.float)
    plot_validation(context, loss, val_loss, target_path)
    
    # save model
    filepath = os.path.join(target_path, name)
    dump(model, open(filepath, 'wb'))
    context.log_artifact(key, target_path=filepath)
        
def plot_validation(
    context: MLClientCtx,
    train_metric,
    valid_metric,
    target_path: str = '',
    name: str = "history.png",
    key: str = 'training-validation-plot'
):
    """Plot train and validation loss curves

    These curves represent the training round losses from the training
    and validation sets.
    
    :param context:         the function context
    :param train_metric:    train metric
    :param valid_metric:    validation metric
    :param target_path:     destinatin path for train/volidation history plot artifact
    """
    # generate plot
    plt.plot(train_metric)
    plt.plot(valid_metric)
    plt.title("training validation results")
    plt.xlabel("epoch")
    plt.ylabel("")
    plt.legend(["train", "valid"])
    fig = plt.gcf()

    # save figure and log artifact
    plotpath = os.path.join(target_path, name)
    plt.savefig(plotpath)
    context.log_artifact(PlotArtifact(key, body=fig))

    # plot cleanup
    plt.cla()
    plt.clf()
    plt.close()        


    
def keras_classifier_generator(
    metrics: list = [],
    input_size: int = 20,
    dropout: float = 0.5,
    output_bias: float = None,
    learning_rate: float = 1e-3
):
    """Generate a super simple classifier

    :param metrics:      select metrics to be evaluated
    :param output_bias:  layer initializer
    :param input_size:   number of features, size of input
    :param dropout:      dropout frequency
    :param learning_rate:

    returns a compiled keras model used as input to the KerasClassifer wrapper
    """
    if output_bias is not None:
        output_bias = Constant(output_bias)

    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=(input_size,)),
            Dropout(dropout),
            Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=metrics
    )

    return model    