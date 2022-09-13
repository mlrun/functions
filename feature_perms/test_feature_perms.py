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
FEATURE_OUTPUT = "feature-importances-permute-tbl"


def arc_to_parquet():
    from mlrun import import_function

    archive_func = import_function('hub://arc_to_parquet')

    archive_run = archive_func.run(
        handler="arc_to_parquet",
        params={"key": "rent", "stats": True, "file_ext": "csv"},
        inputs={"archive_url": DATA_URL},
        artifact_path=os.getcwd() + '/artifacts',
        local=True,
    )

    return archive_run.artifact('rent').url


def sklearn_classifier(run):
    cwd = os.getcwd()
    file_path = str(Path(cwd).parent.absolute()) + "/sklearn_classifier/sklearn_classifier.py"
    fn = code_to_function(
        name='test_sklearn_classifier',
        filename=file_path,
        handler="train_model",
        kind="local",
    )

    fn.spec.command = file_path
    fn.run(
        params={
            "sample": -5_000,  # 5k random rows,
            "model_pkg_class": "sklearn.ensemble.RandomForestClassifier",
            "label_column": "interest_level",
            "CLASS_n_estimators": 100,
            "CLASS_min_samples_leaf": 1,
            "CLASS_n_jobs": -1,
            "CLASS_oob_score": True,
        },
        handler="train_model",
        inputs={"dataset": run.outputs["rent"]},
        artifact_path='artifacts',
    )


def train_model(data):
    from mlrun import import_function

    train = import_function('hub://sklearn_classifier')

    train_run = train.run(
        inputs={"dataset": data},
        params={
            "sample": -5_000,  # 5k random rows,
            "model_pkg_class": "sklearn.ensemble.RandomForestClassifier",
            "label_column": "interest_level",
            "CLASS_n_estimators": 100,
            "CLASS_min_samples_leaf": 1,
            "CLASS_n_jobs": -1,
            "CLASS_oob_score": True,
        },
        local=True
    )

    return train_run.artifact('model').url


def test_feature_selection_run_local():
    data = arc_to_parquet()
    model = train_model(data)
    labels = "interest_level"
    fn = code_to_function(
        name='test_run_local_feature_perms',
        filename="feature_perms.py",
        handler="permutation_importance",
        kind="local",
    )
    fn.spec.command = "feature_perms.py"

    run = fn.run(
        params={
            "labels": labels,
            "plots_dest": "plots",
        },
        inputs={
            "model": model,
            "dataset": data,
        },
        artifact_path='artifacts',
    )

    assert run.artifact(FEATURE_OUTPUT).get()


def test_feature_perms_import_function():
    data = arc_to_parquet()
    model = train_model(data)
    labels = "interest_level"
    fn = import_function("function.yaml")

    run = fn.run(
        params={
            "labels": labels,
            "plots_dest": "plots"
        },
        inputs={
            "model": model,
            "dataset": data},
        artifact_path=os.getcwd() + '/artifacts',
        local=True,
    )

    assert run.artifact(FEATURE_OUTPUT).get()
