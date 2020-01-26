import pandas as pd
import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from cloudpickle import dump

import pyarrow.parquet as pq
import pyarrow as pa

from sklearn.model_selection import train_test_split
from typing import Optional, Union
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_valid_test_splitter(
    context: Optional[MLClientCtx] = None,
    src_file: Union[DataItem, str] = '',
    header: Union[DataItem, str, list] = '',
    sample: int = -1,
    label_column: str = 'labels',
    test_size: float = 0.1,
    train_val_split: float = 0.75,
    target_path: str = '',
    name: str = '',
    key: str = '',
    random_state = 1
) -> None:
    """Split raw data input into train, validation and test sets.

    :param context:         the function context
    :param src_file:        ('raw') name of raw data file
    :param header:          (None) header artifact or list of column names.
    :param sample:          (-1). Selects the first n rows, or select a sample starting
                            from the first. If negative <-1, select a random sample from 
                            the entire file
    :param label_column:    ground-truth (y) labels
    :param test_size:       (0.1) test set size
    :param train_val_split: (0.75) Once the test set has been removed the 
                            training set gets this proportion.
    :param target_path:     folder location of files
    :param name:            destination prefix name for model files
    :param key:             key for model artifact
    :param random_state:    (1) sklearn rng seed
    """
    srcfilepath = os.path.join(target_path, str(src_file))

    if (sample == -1) or (sample >= 1):
        # get all rows, or contiguous sample starting at row 1.
        raw = pq.read_table(srcfilepath).to_pandas()
        labels = raw.pop(label_column)
        raw = raw.iloc[:sample, :]
        labels = labels.iloc[:sample]
    else:
        # grab a random sample
        #raw = pd.read_parquet(srcfilepath, engine='pyarrow').sample(sample*-1)
        raw = pq.read_table(srcfilepath).to_pandas().sample(sample*-1)
        labels = raw.pop(label_column)
    
    # double split tp generate 3 data sets: train, validation and test
    x, xtest, y, ytest = train_test_split(raw, labels, test_size=test_size, 
                                          random_state=random_state)
   
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, 
                                                      train_size=train_val_split, 
                                                      random_state=random_state)        

    if name:
        name = '-' + name
    
    # save header
    f = os.path.join(target_path, name + 'header.pkl')
    dump(raw.columns.values, open(f, 'wb'))
    context.log_artifact('header', target_path=f)
    
    # save data sets
    f = os.path.join(target_path, name + 'xtrain.pqt')
    xtrain.to_parquet(f)
    context.log_artifact('xtrain', target_path=f)
    
    f = os.path.join(target_path, name + 'xvalid.pqt')
    xvalid.to_parquet(f)
    context.log_artifact('xvalid', target_path=f)
    
    f = os.path.join(target_path, name + 'xtest.pqt')
    xtest.to_parquet(f)
    context.log_artifact('xtest', target_path=f)
    
    f = os.path.join(target_path, name + 'ytrain.pqt')
    pd.DataFrame({'labels': ytrain}).to_parquet(f)
    context.log_artifact('ytrain', target_path=f)
    
    f = os.path.join(target_path, name + 'yvalid.pqt')
    pd.DataFrame({'labels': yvalid}).to_parquet(f)
    context.log_artifact('yvalid', target_path=f)
    
    f = os.path.join(target_path, name + 'ytest.pqt')
    pd.DataFrame({'labels': ytest}).to_parquet(f)
    context.log_artifact('ytest', target_path=f)
