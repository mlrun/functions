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
import pickle
import pandas as pd


def generate_data():
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


def test_import_sklearn_classifier():
    acquire_run = generate_data()
    fn = mlrun.import_function("function.yaml")
    # define model
    params = {"model_pkg_class": "sklearn.ensemble.RandomForestClassifier",
              "label_column": "labels"}

    train_run = fn.run(params=params,
                       inputs={"dataset": acquire_run.status.artifacts[0]['spec']['target_path']},
                       local=True,
                       artifact_path="./")

    for artifact in train_run.status.artifacts:
        if artifact['kind'] == 'model':
            assert os.path.exists(artifact['spec']['target_path']), 'Could not find model dir'
            break

    assert os.path.exists(train_run.status.artifacts[0]['spec']['target_path'])
    model = pickle.load(open(artifact['spec']['target_path'] + artifact['spec']['model_file'], 'rb'))
    df = pd.read_csv(acquire_run.status.artifacts[0]['spec']['target_path'])
    x = df.drop(['labels'], axis=1).iloc[0:1]
    y_true = df['labels'][0]
    y_pred = model.predict_proba(x).argmax()
    assert y_pred == y_true, "Failed to predict correctly"
