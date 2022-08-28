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
from pathlib import Path
import os

ARTIFACTS_PATH = 'artifacts'
DATA_URL = "https://raw.githubusercontent.com/parrt/random-forest-importances/master/notebooks/data/rent.csv"
FEATURE_OUTPUT = ARTIFACTS_PATH + "/feature-importances-permute-tbl.parquet"


def arc_to_parquet():
    from mlrun import import_function
    from mlrun.platforms import auto_mount

    archive_func = import_function('hub://arc_to_parquet')
    archive_run = archive_func.run(handler="arc_to_parquet",
                                   params={"key": "rent", "stats": True, "file_ext": "csv"},
                                   inputs={"archive_url": DATA_URL},
                                   artifact_path=os.getcwd() + '/artifacts'
                                   , local=True
                                   )


def sklearn_classifier(run):
    cwd = os.getcwd()
    file_path = str(Path(cwd).parent.absolute()) + "/sklearn_classifier/sklearn_classifier.py"
    fn = code_to_function(name='test_sklearn_classifier',
                          filename=file_path,
                          handler="train_model",
                          kind="local",
                          )
    fn.spec.command = file_path
    fn.run(params={
        "sample": -5_000,  # 5k random rows,
        "model_pkg_class": "sklearn.ensemble.RandomForestClassifier",
        "label_column": "interest_level",
        "CLASS_n_estimators": 100,
        "CLASS_min_samples_leaf": 1,
        "CLASS_n_jobs": -1,
        "CLASS_oob_score": True},
        handler="train_model",
        inputs={"dataset": run.outputs["rent"]},
        artifact_path='artifacts'
        # , local=True
    )


def train_model():
    from mlrun import import_function
    from mlrun.platforms import auto_mount

    train = import_function('hub://sklearn_classifier')
    # .apply(auto_mount())

    train_run = train.run(
        inputs={"dataset": "artifacts/rent.csv"},
        params={
            "sample": -5_000,  # 5k random rows,
            "model_pkg_class": "sklearn.ensemble.RandomForestClassifier",
            "label_column": "interest_level",
            "CLASS_n_estimators": 100,
            "CLASS_min_samples_leaf": 1,
            "CLASS_n_jobs": -1,
            "CLASS_oob_score": True},
        local=True)


def test_feature_selection_run_local():
    arc_to_parquet()
    train_model()
    data = "artifacts/rent.csv"
    labels = "interest_level"
    model = "model/model.pkl"
    fn = code_to_function(name='test_run_local_feature_perms',
                          filename="feature_perms.py",
                          handler="permutation_importance",
                          kind="local",
                          )
    fn.spec.command = "feature_perms.py"
    fi_perms = fn.run(params={"labels": labels,
                              "plots_dest": "plots"},
                      inputs={"model": model, "dataset": data},
                      artifact_path='artifacts')
    assert Path(FEATURE_OUTPUT).is_file()


def test_feature_perms_import_function():
    arc_to_parquet()
    train_model()
    data = "artifacts/rent.csv"
    labels = "interest_level"
    model = "model/model.pkl"
    fi_perms = import_function("function.yaml")
    fi_perms.run(params={"labels": labels,
                         "plots_dest": "plots"},
                 inputs={"model": model, "dataset": data},
                 artifact_path=os.getcwd() + '/artifacts'
                 , local=True)
    assert Path(FEATURE_OUTPUT).is_file()
