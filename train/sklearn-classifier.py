from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
import pandas as pd
import lightgbm as lgb
from typing import Optional, Union
import os
from sklearn.model_selection import train_test_split
import importlib
from cloudpickle import dump

def train(
    context: Optional[MLClientCtx] = None,
    src_file: Union[DataItem, str] = '',
    SKClassifier: str  = '',
    callbacks  = [],
    test_size: float = 0.1,
    train_val_split: float = 0.75,
    sample: int = -1,
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
    :param src_file:        ('raw') name of raw data file
    :param sample:          (-1). Selects the first n rows, or select a sample starting
                            from the first. If negative <-1, select a random sample from 
                            the entire file
    :param header:          (None) header artifact or list of column names.
    :param SKClassifier:    string module and classname of classifier
    :param callbacks
    :param test_size:       (0.1) test set size
    :param train_val_split: (0.75) Once the test set has been removed the 
                            training set gets this proportion.
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
    if isinstance(src_file, DataItem):
        src_file = str(src_file)
    srcfilepath = os.path.join(target_path, src_file)

    # save only a sample, intended for debugging
    if (sample == -1) or (sample >= 1):
        # get all rows, or contiguous sample starting at row 1.
        raw = pd.read_parquet(srcfilepath, engine='pyarrow')
        labels = raw.pop('labels')
        raw = raw.iloc[:sample, :]
        labels = labels.iloc[:sample]
    else:
        # grab a random sample
        raw = pd.read_parquet(srcfilepath, engine='pyarrow').sample(sample*-1)
        labels = raw.pop('labels')
    
    # double split tp generate 3 data sets: train, validation and test
    x, xtest, y, ytest = train_test_split(raw, labels, train_size=1-test_size, 
                                          random_state=random_state)
   
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, 
                                                      train_size=train_val_split, 
                                                      random_state=random_state)        
   
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
     
    context.log_result("train_accuracy", float(clf.score(xtrain, ytrain)))
    
    # plot train and validation history, save and log
    loss = np.asarray(model.evals_result_['train']['binary_logloss'], dtype=np.float)
    val_loss = np.asarray(model.evals_result_['valid']['binary_logloss'], dtype=np.float)
    plot_validation(loss, val_loss)
    
    # save model
    filepath = os.path.join(target_path, name)
    dump(clf, open(filepath, 'wb'))
    context.log_artifact(key, target_path=filepath) #, labels=exp_labels)
    # save test data
    for t in ['x', 'y']:
        fname = t + 'test.pkl'
        filepath = os.path.join(target_path, fname)
        dump(xtest, open(filepath, 'wb'))
        context.log_artifact(t+'test', target_path=filepath)
        
        
def plot_validation(train_metric, valid_metric):
    """Plot train and validation loss curves

    These curves represent the training round losses from the training
    and validation sets.
    
    :param train_metric:    train metric
    :param valid_metric:    validation metric
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
    plotpath = path.join(target_path, "history.png")
    plt.savefig(plotpath)
    context.log_artifact(PlotArtifact('training-validation-plot', body=fig, target_path=plotpath))

    # plot cleanup
    plt.cla()
    plt.clf()
    plt.close()        
