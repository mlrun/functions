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
from mlrun import code_to_function, import_function
import os
import pandas as pd


def get_class_data():
    fn = import_function("hub://gen_class_data")
    run = fn.run(
        params={
            "n_samples": 10_000,
            "m_features": 5,
            "k_classes": 2,
            "header": None,
            "weight": [0.5, 0.5],
            "sk_params": {"n_informative": 2},
            "file_ext": "csv",
        },
        local=True,
        artifact_path="./artifacts/inputs",
    )
    return run.artifact('classifier-data').url


def xgb_trainer():
    data = get_class_data()
    fn = code_to_function(
        name='xgb_trainer',
        filename=os.path.dirname(os.path.dirname(__file__)) + "/xgb_trainer/xgb_trainer.py",
        handler="train_model",
        kind="job",
    )
    run = fn.run(
        params={
            "model_type": "classifier",
            "CLASS_tree_method": "hist",
            "CLASS_objective": "binary:logistic",
            "CLASS_booster": "gbtree",
            "FIT_verbose": 0,
            "label_column": "labels",
            "test_set": "./artifacts/test-set",
        },
        local=True,
        inputs={"dataset": data},
    )

    return data, run.artifact('model').url


def test_xgb_test_code_to_function():
    data, model = xgb_trainer()
    fn = code_to_function(
        name='test_xgb_test',
        filename=os.path.dirname(os.path.dirname(__file__)) + "/xgb_test/xgb_test.py",
        handler="xgb_test",
        kind="job",
    )
    run = fn.run(
        params={
            "label_column": "labels",
            "plots_dest": "plots/xgb_test",
        },
        local=True,
        inputs={
            "test_set": data,
            "models_path": model,
        }
    )

    assert run.outputs['accuracy'] and run.state() == 'completed'


def test_local_xgb_test_import_local_function():
    # importing data preparation function (gen_class_data) locally
    fn = import_function("hub://gen_class_data")
    run = fn.run(
        params={
            "n_samples": 10_000,
            "m_features": 5,
            "k_classes": 2,
            "header": None,
            "weight": [0.5, 0.5],
            "sk_params": {"n_informative": 2},
            "file_ext": "csv",
        },
        local=True,
        artifact_path="./artifacts/inputs",
    )
    data = run.artifact('classifier-data').url

    # importing model training function (xgb_trainer) locally
    fn = import_function("../xgb_trainer/function.yaml")
    run = fn.run(
        params={
            "model_type": "classifier",
            "CLASS_tree_method": "hist",
            "CLASS_objective": "binary:logistic",
            "CLASS_booster": "gbtree",
            "FIT_verbose": 0,
            "label_column": "labels",
            "test_set": "./artifacts/test-set",
        },
        local=True,
        inputs={"dataset": data},
    )
    model = run.artifact('model').url

    # importing xgb_test function.yaml and running tests
    fn = import_function("function.yaml")
    run = fn.run(
        params={
            "label_column": "labels",
            "plots_dest": "plots/xgb_test",
        },
        local=True,
        inputs={
            "test_set": data,
            "models_path": model,
        }
    )

    # tests for gen_class_data
    assert data
    df = pd.read_csv(data)
    assert (True if df["labels"].sum() == 5008 else False)
    # tests for xgb_trainer
    assert model
    # no tests for xgb_test (it is a test already)
