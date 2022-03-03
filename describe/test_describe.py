import os
import shutil
from pathlib import Path

import pytest
from mlrun import import_function
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import mlrun

from describe import summarize
from typing import Set

DATA_URL = "https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv"
PLOTS_PATH = os.path.abspath("./artifacts/plots")
ARTIFACTS_PATH = os.path.abspath("./artifacts")


PARAM = [{"name": "small_data", "n_samples": 100, "n_features": 5, "n_classes": 3, "n_informative": 3},
         {"name": "big_amount_of_samples", "n_samples": 10000, "n_features": 5, "n_classes": 3, "n_informative": 3},
         {"name": "big_amount_of_features", "n_samples": 100, "n_features": 20, "n_classes": 3, "n_informative": 3},
         {"name": "big_data", "n_samples": 10000, "n_features": 20, "n_classes": 4, "n_informative": 5},
         ]


def _validate_paths(paths: Set):
    base_folder = PLOTS_PATH
    for path in paths:
        full_path = os.path.join(base_folder, path)
        if Path(full_path).is_file():
            print("File exist")
        else:
            raise FileNotFoundError


@pytest.fixture(autouse=True)
def run_around_tests():
    if Path(PLOTS_PATH).is_dir():
        shutil.rmtree(PLOTS_PATH)
    if Path(ARTIFACTS_PATH).is_dir():
        shutil.rmtree(ARTIFACTS_PATH)
    os.mkdir(ARTIFACTS_PATH)
    yield
    _validate_paths(
        {
            "corr.html",
            "correlation-matrix.csv",
            "hist.html",
            "imbalance.html",
            "imbalance-weights-vec.csv",
            "violin.html",
        }
    )


def test_sanity_local():
    # Setting environment:
    describe_func = import_function("function.yaml")

    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler='summarize',
            inputs={"table": DATA_URL},
            params={"update_dataset": True, "label_column": "label"},
            artifact_path=os.path.abspath("./artifacts"),
            local=True
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")


@pytest.mark.parametrize("param", PARAM)
def test_different_size_of_dataset(param):
    X, y = make_classification(n_samples=param['n_samples'], n_features=param['n_features'],
                               n_classes=param['n_classes'], n_informative=param['n_informative'],
                               random_state=18, class_sep=2,)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(param['n_features'])])
    df['label'] = y
    data_path = "artifacts/random_dataset.parquet"
    df.to_parquet("artifacts/random_dataset.parquet")

    describe_func = import_function("function.yaml")

    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler='summarize',
            inputs={"table": data_path},
            params={"update_dataset": True, "label_column": "label"},
            artifact_path=os.path.abspath("./artifacts"),
            local=True
        )
    except Exception as exception:
        print(f"- {param['name']} test failed - raised the following error:\n- {exception}")