import os
import shutil
from pathlib import Path
from typing import Set

import mlrun
import pandas as pd
import pytest
from mlrun import import_function
from mlrun.execution import MLClientCtx
from sklearn.datasets import make_classification

DATA_PATH = os.path.abspath("./artifacts/random_dataset.parquet")
PLOTS_PATH = os.path.abspath("./artifacts/plots")
ARTIFACTS_PATH = os.path.abspath("./artifacts")


def _validate_paths(paths: Set):
    """
    Check if all the expected plot are saved
    """
    base_folder = PLOTS_PATH
    for path in paths:
        full_path = os.path.join(base_folder, path)
        if Path(full_path).is_file():
            print(f"{path} exist")
        else:
            assert FileNotFoundError(f"{path} not found!")


@pytest.fixture(autouse=True)
def run_around_tests():
    """
    Wraps the tests and resets the local paths + activate _validate_paths after each test
    """
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
            "hist_mat.html",
            "imbalance.html",
            "imbalance-weights-vec.csv",
            "violin.html",
        }
    )


def test_sanity_local():
    """
    Test simple scenario
    """
    describe_func = import_function("function.yaml")
    is_test_passed = True
    _creat_data(n_samples=100, n_features=5, n_classes=3, n_informative=3)
    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analysis",
            inputs={"table": DATA_PATH},
            params={"update_dataset": True, "label_column": "label"},
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False
    assert is_test_passed


@pytest.mark.parametrize("n_samples", [100, 1000, 5000, 10000])
@pytest.mark.parametrize("n_features", [5, 10, 20])
@pytest.mark.parametrize("n_classes", [3, 4])
@pytest.mark.parametrize("n_informative", [3])
def test_different_size_of_dataset(n_samples, n_features, n_classes, n_informative):
    """
    Test different size of data
    """
    is_test_passed = True
    df = _creat_data(n_samples, n_features, n_classes, n_informative)
    describe_func = import_function("function.yaml")

    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analysis",
            inputs={"table": DATA_PATH},
            params={"update_dataset": True, "label_column": "label"},
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )
    except Exception as exception:
        print(f" The test failed - raised the following error:\n- {exception}")
        is_test_passed = False

    _validate_paths({f"hist_{col}.html" for col in df.columns if col is not "label"})
    assert is_test_passed


def test_data_already_loaded():
    """
    Test scenario on already loaded data artifact
    """

    log_data_function = mlrun.code_to_function(
        filename="test_describe.py",
        name="log_data",
        kind="job",
        image="mlrun/ml-models",
    )
    df = _creat_data(n_samples=100, n_features=5, n_classes=3, n_informative=3)
    log_data_run = log_data_function.run(
        handler="_log_data",
        params={"table": DATA_PATH},
        local=True,
    )
    describe_func = import_function("function.yaml")
    is_test_passed = True
    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analysis",
            inputs={"table": log_data_run.outputs["dataset"]},
            params={"update_dataset": True, "label_column": "label"},
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False

    created_artifact = [*describe_run.outputs.keys()]

    expected_artifacts = [
        f"histogram_{col}" for col in df.columns if col is not "label"
    ] + [
        "correlation-matrix csv",
        "correlation-matrix",
        "histograms matrix",
        "imbalance",
        "imbalance-weights-vec",
        "violin",
    ]
    for artifact in expected_artifacts:
        if artifact not in created_artifact:
            print(f"{artifact} artifact not found!")
            is_test_passed = False

            break
    assert is_test_passed


def _log_data(context: MLClientCtx, table: str):
    """
    Log the data using context.log_dataset
    """
    df = pd.read_parquet(table)
    context.log_dataset(key="dataset", db_key="dataset", stats=True, df=df)


def _creat_data(n_samples, n_features, n_classes, n_informative):
    """
    Create df and save it as artifacts/random_dataset.parquet
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        random_state=18,
        class_sep=2,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = y
    df.to_parquet("artifacts/random_dataset.parquet")
    return df
