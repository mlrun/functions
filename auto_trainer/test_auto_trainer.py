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
import os
import tempfile
from typing import Tuple

import mlrun
import pandas as pd
import pytest
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

MODELS = [
    ("sklearn.linear_model.LinearRegression", "regression"),
    ("sklearn.ensemble.RandomForestClassifier", "classification"),
    ("xgboost.XGBRegressor", "regression"),
    ("lightgbm.LGBMClassifier", "classification"),
]

REQUIRED_ENV_VARS = [
    "MLRUN_DBPATH",
    "MLRUN_ARTIFACT_PATH",
    "V3IO_USERNAME",
    "V3IO_API",
    "V3IO_ACCESS_KEY",
]


def _validate_environment_variables() -> bool:
    """
    Checks that all required Environment variables are set.
    """
    environment_keys = os.environ.keys()
    return all(key in environment_keys for key in REQUIRED_ENV_VARS)


def _get_dataset(problem_type: str, filepath: str = ".", n_classes: int = 2):
    if problem_type == "classification":
        x, y = make_classification(n_classes=n_classes)
    elif problem_type == "regression":
        x, y = make_regression(n_targets=1)
    elif problem_type == "multilabel_classification":
        x, y = make_multilabel_classification(n_classes=n_classes)
    else:
        raise ValueError(f"Not supporting problem type = {problem_type}")

    features = [f"f_{i}" for i in range(x.shape[1])]
    if y.ndim == 1:
        labels = ["labels"]
    else:
        labels = [f"label_{i}" for i in range(y.shape[1])]
    dataset = pd.concat(
        [pd.DataFrame(x, columns=features), pd.DataFrame(y, columns=labels)], axis=1
    )
    filename = f"{filepath}/{problem_type}_dataset.csv"
    dataset.to_csv(filename, index=False)
    return filename, labels


def _assert_train_handler(train_run):
    assert train_run and all(
        key in train_run.outputs for key in ["model", "test_set"]
    ), "outputs should include more data"


@pytest.mark.parametrize("model", MODELS)
def test_train(model: Tuple[str, str]):
    dataset, label_columns = _get_dataset(model[1])
    is_test_passed = True
    # Importing function:
    # fn = import_function("function.yaml")
    project = mlrun.new_project("auto-trainer-test", context="./")
    fn = project.set_function("function.yaml", "train", kind="job", image="mlrun/mlrun")
    # fn = mlrun.code_to_function("train", kind="job", filename="function.yaml", image="mlrun/mlrun", handler="train")
    train_run = None
    model_name = model[0].split(".")[-1]
    labels = {"label1": "my-value"}
    try:
        train_run = fn.run(
            inputs={"dataset": dataset},
            params={
                "drop_columns": ["f_0", "f_2"],
                "model_class": model[0],
                "model_name": f"model_{model_name}",
                "label_columns": label_columns,
                "train_test_split_size": 0.2,
                "labels": labels,
            },
            handler="train",
            local=True,
        )
    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False

    assert is_test_passed, "The test failed"
    _assert_train_handler(train_run)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_train_evaluate(model: Tuple[str, str]):
    dataset, label_columns = _get_dataset(model[1])
    is_test_passed = True
    # Importing function:
    project = mlrun.new_project("auto-trainer-test", context="./")
    fn = project.set_function("function.yaml", "train", kind="job", image="mlrun/mlrun")
    temp_dir = tempfile.mkdtemp()

    evaluate_run = None
    model_name = model[0].split(".")[-1]
    try:
        train_run = fn.run(
            inputs={"dataset": dataset},
            params={
                "drop_columns": ["f_0", "f_2"],
                "model_class": model[0],
                "model_name": f"model_{model_name}",
                "label_columns": label_columns,
                "train_test_split_size": 0.2,
            },
            handler="train",
            local=True,
            artifact_path=temp_dir,
        )
        _assert_train_handler(train_run)

        evaluate_run = fn.run(
            inputs={"dataset": train_run.outputs["test_set"]},
            params={
                "model": train_run.outputs["model"],
                "label_columns": label_columns,
            },
            handler="evaluate",
            local=True,
            artifact_path=temp_dir,
        )
    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False

    assert is_test_passed, "The test failed"
    assert (
        evaluate_run and "evaluation-test_set" in evaluate_run.outputs
    ), "Missing fields in evaluate_run"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_train_predict(model: Tuple[str, str]):
    is_test_passed = True
    dataset, label_columns = _get_dataset(model[1])
    df = pd.read_csv(dataset)
    sample = df.head().drop("labels", axis=1).values.tolist()
    # Importing function:
    project = mlrun.new_project("auto-trainer-test", context="./")
    fn = project.set_function("function.yaml", "train", kind="job", image="mlrun/mlrun")
    temp_dir = tempfile.mkdtemp()

    predict_run = None
    model_name = model[0].split(".")[-1]
    try:
        train_run = fn.run(
            inputs={"dataset": dataset},
            params={
                "drop_columns": ["f_0", "f_2"],
                "model_class": model[0],
                "model_name": f"model_{model_name}",
                "label_columns": label_columns,
                "train_test_split_size": 0.2,
            },
            handler="train",
            local=True,
            artifact_path=temp_dir,
        )
        _assert_train_handler(train_run)

        predict_run = fn.run(
            params={
                "dataset": sample,
                "drop_columns": [0, 2],
                "model": train_run.outputs["model"],
                "label_columns": label_columns,
            },
            handler="predict",
            local=True,
            artifact_path=temp_dir,
        )
    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False

    assert is_test_passed, "The test failed"
    assert (
        predict_run and "prediction" in predict_run.outputs
    ), "Prediction field must be in the output"
