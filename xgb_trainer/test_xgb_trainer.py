# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from mlrun import code_to_function, import_function, DataItem
from mlrun.execution import MLClientCtx
import os


def get_class_data():
    fn = import_function('hub://gen_class_data')
    run = fn.run(params={'key': 'classifier-data',
                         'n_samples': 10_000,
                         'm_features': 5,
                         'k_classes': 2,
                         'header': None,
                         'weight': [0.5, 0.5],
                         'sk_params': {'n_informative': 2},
                         'file_ext': 'csv'}, local=True, artifact_path='./')
    return run

def test_xgb_trainer_code_to_function():
    gen_data_run = get_class_data()
    fn = code_to_function(name='test_xgb_trainer',
                          filename='xgb_trainer.py',
                          handler='train_model',
                          kind='local')
    
    run = fn.run(params={'model_type': 'classifier',
                         'CLASS_tree_method': 'hist',
                         'CLASS_objective': 'binary:logistic',
                         'CLASS_booster': 'gbtree',
                         'FIT_verbose': 0,
                         'label_column': 'labels',
                         'test_set': './'},
                 local=False,
                 inputs={'dataset': gen_data_run.artifact('classifier-data').url})

    assert (run.artifact('model'))


def test_local_xgb_trainer_import_function():
    # importing data preparation function locally
    gen_data_run = get_class_data()

    fn = import_function('function.yaml')
    run = fn.run(params={'model_type': 'classifier',
                         'CLASS_tree_method': 'hist',
                         'CLASS_objective': 'binary:logistic',
                         'CLASS_booster': 'gbtree',
                         'FIT_verbose': 0,
                         'label_column': 'labels',
                         'test_set': './'},
                 local=True, inputs={'dataset': gen_data_run.artifact('classifier-data').url})

    assert (run.artifact('model'))