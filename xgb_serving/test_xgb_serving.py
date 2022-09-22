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
from mlrun import import_function
import os
import pandas as pd
from xgb_serving import XGBoostModel

def test_local_xgb_serving():
    # importing data preparation function (gen_class_data) locally
    fn = import_function("hub://gen_class_data")
    gen_data_run = fn.run(params={"n_samples": 10_000,
                                  "m_features": 5,
                                  "k_classes": 2,
                                  "header": None,
                                  "weight": [0.5, 0.5],
                                  "sk_params": {"n_informative": 2},
                                  "file_ext": "csv"},
                          local=True,
                          artifact_path="./")

    # importing model training function (xgb_trainer) locally
    fn = import_function("../xgb_trainer/function.yaml")
    xgb_trainer_run = fn.run(params={"model_type": "classifier",
                                     "CLASS_tree_method": "hist",
                                     "CLASS_objective": "binary:logistic",
                                     "CLASS_booster": "gbtree",
                                     "FIT_verbose": 0,
                                     "label_column": "labels",
                                     "test_set": "./"},
                             local=True,
                             inputs={"dataset": gen_data_run.artifact('classifier-data').url},
                             artifact_path='./')

    # because this class is implemented with MLModelServer, creating a class instance and not to_mock_server(V2_Model_Server).
    model = xgb_trainer_run.artifact('model').url
    my_server = XGBoostModel("my-model", model_dir=model)
    my_server.load()
    # Testing the model
    xtest = pd.read_csv(gen_data_run.artifact('classifier-data').url)
    preds = my_server.predict({"instances": xtest.values[:10, :-1].tolist()})
    assert (True if preds == [1, 0, 0, 0, 0, 0, 1, 1, 0, 1] else False) is True