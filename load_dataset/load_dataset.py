# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cloudpickle import dump, load

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from typing import IO, AnyStr, Union, List, Optional

def get_toy_data(
    context: MLClientCtx,
    dataset: str,
    params: dict = {}
) -> None:
    """Loads a scikit-learn toy dataset for classification or regression
    
    The following datasets are available ('name' : desription):
    
        'boston'          : boston house-prices dataset (regression)
        'iris'            : iris dataset (classification)
        'diabetes'        : diabetes dataset (regression)
        'digits'          : digits dataset (classification)
        'linnerud'        : linnerud dataset (multivariate regression)
        'wine'            : wine dataset (classification)
        'breast_cancer'   : breast cancer wisconsin dataset (classification)
    
    The scikit-learn functions return a data bunch including the following items:
    - data              the features matrix
    - target            the ground truth labels
    - DESCR             a description of the dataset
    - feature_names     header for data
    
    The features (and their names) are stored with the target labels in a DataFrame.

    For further details see https://scikit-learn.org/stable/datasets/index.html#toy-datasets
    
    :param context:    function execution context
    :param dataset:    name of the dataset to load 
    :param params:     params of the sklearn load_data method
    """
    filepath = os.path.join(context.artifact_path, dataset) + '.pqt'
    
    # check to see if we haven't already downloaded the file
    if not os.path.isfile(filepath):
        artifact_path = context.artifact_path

        # reach into module and import the appropriate load_xxx function
        pkg_module = 'sklearn.datasets'
        fname = f'load_{dataset}'

        pkg_module = __import__(pkg_module, fromlist=[fname])
        load_data_fn = getattr(pkg_module, fname)
        
        data = load_data_fn(**params)
        feature_names = data['feature_names']

        # save
        xy = np.concatenate([data['data'], data['target'].reshape(-1, 1)], axis=1)
        if hasattr(feature_names, 'append'):
            # its a list
            feature_names.append('labels')
        else:
            # its an array
            feature_names = np.append(feature_names, 'labels')
        df = pd.DataFrame(data=xy, columns=feature_names)
        df.to_parquet(filepath, engine='pyarrow', index=False)
        
    # either we just downloaded file, or it exists, log it:
    context.log_artifact(dataset, local_path=filepath.split('/')[-1])
        