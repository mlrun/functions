import os
import shutil
from pathlib import Path
from typing import Set

import mlrun
import pandas as pd
import pytest
from mlrun import code_to_function, import_function, new_function
from mlrun.execution import MLClientCtx
from sklearn.datasets import make_classification, make_regression

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
            raise FileNotFoundError(f"{path} not found!")


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
            "correlation.html",
            "correlation-matrix.csv",
            "scatter-2d.html",
            "violin.html",
            "describe.csv",
            # "hist.html",
            "histograms.html",
        }
    )


@pytest.mark.parametrize("problem_type", ["classification"])
def test_sanity_local(problem_type):
    """
    Test simple scenario
    """
    describe_func = import_function("function.yaml")
    is_test_passed = True
    _create_data(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        n_informative=3,
        reg=(problem_type == "regression"),
    )
    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analyze",
            inputs={"table": DATA_PATH},
            params={"label_column": "label", "problem_type": problem_type},
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False
    _validate_paths(
        {
            "imbalance.html",
            "imbalance-weights-vec.csv",
        }
    )

    assert is_test_passed


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_none_label(problem_type):
    """
    Test simple scenario
    """
    describe_func = import_function("function.yaml")
    is_test_passed = True
    _create_data(
        n_samples=100,
        n_features=5,
        n_classes=3,
        n_informative=3,
        reg=(problem_type == "regression"),
    )
    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analyze",
            inputs={"table": DATA_PATH},
            params={"label_column": "", "problem_type": problem_type},
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
@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_different_size_of_dataset(
    problem_type, n_samples, n_features, n_classes, n_informative
):
    """
    Test different size of data
    """
    is_test_passed = True
    df = _create_data(
        n_samples,
        n_features,
        n_classes,
        n_informative,
        reg=(problem_type == "regression"),
    )
    describe_func = import_function("function.yaml")

    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analyze",
            inputs={"table": DATA_PATH},
            params={"label_column": "label", "problem_type": problem_type},
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )
    except Exception as exception:
        print(f" The test failed - raised the following error:\n- {exception}")
        is_test_passed = False

    _validate_paths(
        {
            "imbalance.html",
            "imbalance-weights-vec.csv",
        }
    )
    assert is_test_passed


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_data_already_loaded(problem_type):
    """
    Test scenario on already loaded data artifact
    """

    log_data_function = mlrun.code_to_function(
        filename="test_describe.py",
        name="log_data",
        kind="job",
        image="mlrun/ml-models",
    )
    df = _create_data(
        n_samples=100,
        n_features=5,
        n_classes=3,
        n_informative=3,
        reg=(problem_type == "regression"),
    )
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
            handler="analyze",
            inputs={"table": log_data_run.outputs["dataset"]},
            params={"label_column": "label", "problem_type": problem_type},
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False

    created_artifact = [*describe_run.outputs.keys()]

    expected_artifacts = [
        "correlation-matrix-csv",
        "correlation",
        "scatter-2d",
        "imbalance",
        "imbalance-weights-vec",
        "violin",
        "histograms",
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


def _create_data(n_samples, n_features, n_classes, n_informative, reg=False):
    """
    Create df and save it as artifacts/random_dataset.parquet
    """
    if not reg:
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
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=18,
        )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = y
    df.to_parquet("artifacts/random_dataset.parquet")
    return df


def _create_dask_func(uri):
    dask_cluster_name = "dask-cluster"
    dask_cluster = new_function(dask_cluster_name, kind="dask", image="mlrun/ml-models")
    dask_cluster.spec.remote = False
    dask_uri = uri
    dask_cluster.export(dask_uri)


def test_import_function_describe_dask():
    dask_uri = "dask_func.yaml"
    _create_dask_func(dask_uri)
    describe_func = import_function("function.yaml")
    is_test_passed = True
    _create_data(n_samples=100, n_features=5, n_classes=3, n_informative=3)
    describe_func.spec.command = "describe_dask.py"

    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analyze",
            inputs={"table": DATA_PATH},
            params={
                "label_column": "label",
                "dask_function": dask_uri,
                "dask_flag": True,
            },
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False
    _validate_paths(
        {
            "imbalance.html",
            "imbalance-weights-vec.csv",
        }
    )
    assert is_test_passed


def test_code_to_function_describe_dask():
    dask_uri = "dask_func.yaml"
    _create_dask_func(dask_uri)
    describe_func = code_to_function(filename="describe.py", kind="local")
    is_test_passed = True
    _create_data(n_samples=100, n_features=5, n_classes=3, n_informative=3)
    describe_func.spec.command = "describe_dask.py"

    try:
        describe_run = describe_func.run(
            name="task-describe",
            handler="analyze",
            inputs={"table": DATA_PATH},
            params={
                "label_column": "label",
                "dask_function": dask_uri,
                "dask_flag": True,
            },
            artifact_path=os.path.abspath("./artifacts"),
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
        is_test_passed = False
    _validate_paths(
        {
            "imbalance.html",
            "imbalance-weights-vec.csv",
        }
    )
    assert is_test_passed
