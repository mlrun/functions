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
from typing import IO, AnyStr, Union, List, Optional


def get_model_config(
    context: MLClientCtx,
    config_url: Union[str, Path, IO[AnyStr]],
) -> None:
    """Retrieve a model config.
    
    A model config consists of a dict with 2 keys:  class_params and
    fit_params and these map to the model's scikit learn API.
    """
    import json
    config = json.load(open(str(config_url), "r"))
    
    os.makedirs(os.path.join(context.artifact_path, 'models'), exist_ok=True)
    context.log_artifact('class_params', body=json.dumps(config['CLASS_PARAMS']), local_path='models/class_params.json') #, labels={'framework': 'xgboost'})
    context.log_artifact('fit_params', body=json.dumps(config['FIT_PARAMS']), local_path='models/fit_params.json') #, labels={'framework': 'xgboost'})
    

def gen_model_config_by_attr(attr: str = 'pedict_proba'):
    """Generate a model config and save as json
    Filters model by attribute `attr`, for example, only classes
    with a `predict_proba` method
    Returns a list of (str, FullArgSpec, FullArgSpec) Tuples, where the first 
    FullArgSpec object contains all model class parameters plus defaults, and 
    the second contains all the fit params plus defaults
    :param attrib:   the attribute filter
    """
    estimators = all_estimators()
    clfs = []
    for name, class_ in estimators:
        if hasattr(class_, 'predict_proba'):
            clfs.append((name, getfullargspec(class_), getfullargspec(class_.fit)))
... etc    