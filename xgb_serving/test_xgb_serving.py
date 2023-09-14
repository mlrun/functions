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
import mlrun
import os
import pandas as pd
from xgb_serving import XGBoostModel


def get_class_data():
    fn = mlrun.import_function('../gen_class_data/function.yaml')
    run = fn.run(params={'key': 'classifier-data',
                         'n_samples': 10_000,
                         'm_features': 5,
                         'k_classes': 2,
                         'header': None,
                         'weight': [0.5, 0.5],
                         'sk_params': {'n_informative': 2},
                         'file_ext': 'csv'}, local=True, artifact_path="./artifacts")
    return run


def xgb_trainer():
    # running data preparation function locally
    gen_data_run = get_class_data()

    fn = mlrun.import_function('../xgb_trainer/function.yaml')
    run = fn.run(params={'model_type': 'classifier',
                         'CLASS_tree_method': 'hist',
                         'CLASS_objective': 'binary:logistic',
                         'CLASS_booster': 'gbtree',
                         'FIT_verbose': 0,
                         'label_column': 'labels'},
                 local=True, inputs={'dataset': gen_data_run.status.artifacts[0]['spec']['target_path']})

    for artifact in run.status.artifacts:
        if artifact['kind'] == 'model':
            assert os.path.exists(artifact['spec']['target_path']), "Failed locating model file"  # validating model exists
            return artifact['spec']['target_path'] + artifact['spec']['model_file'], gen_data_run.status.artifacts[0]['spec']['target_path']
    assert False, "Failed creating model"


def test_local_xgb_serving():
    model_path, dataset_path = xgb_trainer()
    fn = mlrun.import_function('function.yaml')

    fn.add_model(key='my_model', model_path=model_path, class_name='XGBoostModel')
    server = fn.to_mock_server()

    # Testing the model
    df = pd.read_csv(dataset_path)
    x = df.drop(['labels'], axis=1).iloc[0].tolist()
    y_true = df['labels'][0]

    y_pred = server.test(path='/v2/models/my_model/predict', body={"inputs": x})['outputs'][0]
    assert y_true == y_pred
