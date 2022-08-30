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
from pathlib import Path

DATA_URL = 'https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv'
CLASSIFIER_PATH = "artifacts/default/latest/arc-to-parquet-arc_to_parquet_price.yaml"

def generate_data():
    data_url = "https://raw.githubusercontent.com/parrt/random-forest-importances/master/notebooks/data/rent.csv"

    fn = import_function("../arc_to_parquet/function.yaml")
    acquire_run = fn.run(params={"key": "price", "stats": True, "file_ext": "csv"}, inputs={"archive_url": data_url},
                         handler="arc_to_parquet",local=True,artifact_path="artifacts")
    return acquire_run


def test_import_sklearn_classifier():
    acquire_run = generate_data()
    fn = import_function("function.yaml")
    # define model
    params = {
            "sample": -5_000,  # 5k random rows,
            "model_pkg_class": "sklearn.ensemble.RandomForestClassifier",
            "label_column": "interest_level",
            "CLASS_n_estimators": 100,
            "CLASS_min_samples_leaf": 1,
            "CLASS_n_jobs": -1,
            "CLASS_oob_score": True}

    train_run = fn.run(params=params, inputs={"dataset": acquire_run.outputs["price"]},local=True,
                       artifact_path="artifacts")
    assert Path(CLASSIFIER_PATH).is_file()


