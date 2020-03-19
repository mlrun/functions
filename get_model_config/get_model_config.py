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
        my_class = pkg_class.split('.')[-1]
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
        model_json[meta_key]['sklearn_version'] = skversion
        model_json[meta_key]['class'] = '.'.join([estimator[3].__module__, estimator[0]])
        model_configs.append(model_json)
    if len(model_configs) == 1:
        try:
            context.log_artifact('model_config', body=json.dumps(config['CLASS']), local_path='models/class_params.json')
        except Exception as e:
            
        return model_configs[0]
    else:
        # log artifact this? wip, not this week
        return model_configs
  