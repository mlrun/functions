import os
import sys
from pathlib import Path
import shutil
import mlrun

import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult

from great_expectations_mlrun import (
    get_default_datasource_config,
    get_default_checkpoint_config,
    get_data_doc_path,
) 


DATA_ASSET_NAME = "iris_dataset"
DATA_PATH = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"
EXPECTATION_SUITE_NAME = "test_suite"
ROOT_DIRECTORY = f"/tmp/great_expectations"
DATASOURCE_NAME = "pandas_datasource"
DATA_CONNECTOR_NAME = "default_runtime_data_connector_name"


def test_get_default_datasource_config():
    datasource_name = "my_datasource"
    data_connector_name = "my_dataconnector"

    expected_datasource_config = {
        "name": f"{datasource_name}",
        "class_name": "Datasource",
        "module_name": "great_expectations.datasource",
        "execution_engine": {
            "module_name": "great_expectations.execution_engine",
            "class_name": "PandasExecutionEngine",
        },
        "data_connectors": {
            f"{data_connector_name}": {
                "class_name": "RuntimeDataConnector",
                "module_name": "great_expectations.datasource.data_connector",
                "batch_identifiers": ["default_identifier_name"],
            },
        },
    }

    assert (
        get_default_datasource_config(
            datasource_name=datasource_name, data_connector_name=data_connector_name
        )
        == expected_datasource_config
    )


def test_get_default_checkpoint_config():
    checkpoint_name = "my_checkpoint"

    expected_checkpoint_config = {
        "name": checkpoint_name,
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S-my-run-name-template",
    }

    assert (
        get_default_checkpoint_config(checkpoint_name=checkpoint_name)
        == expected_checkpoint_config
    )


def set_expectations(fail=False):
    ge_context = BaseDataContext(
        project_config=DataContextConfig(
            store_backend_defaults=FilesystemStoreBackendDefaults(
                root_directory=ROOT_DIRECTORY
            )
        )
    )

    datasource_config = {
        "name": f"{DATASOURCE_NAME}",
        "class_name": "Datasource",
        "module_name": "great_expectations.datasource",
        "execution_engine": {
            "module_name": "great_expectations.execution_engine",
            "class_name": "PandasExecutionEngine",
        },
        "data_connectors": {
            f"{DATA_CONNECTOR_NAME}": {
                "class_name": "RuntimeDataConnector",
                "module_name": "great_expectations.datasource.data_connector",
                "batch_identifiers": ["default_identifier_name"],
            },
        },
    }
    ge_context.add_datasource(**datasource_config)

    ge_context.create_expectation_suite(
        expectation_suite_name=EXPECTATION_SUITE_NAME, overwrite_existing=True
    )

    df = pd.read_csv(DATA_PATH)

    batch_request = RuntimeBatchRequest(
        datasource_name=DATASOURCE_NAME,
        data_connector_name=DATA_CONNECTOR_NAME,
        data_asset_name=DATA_ASSET_NAME,
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_identifier"},
    )

    validator = ge_context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=EXPECTATION_SUITE_NAME,
    )

    validator.expect_column_values_to_not_be_null(column="sepal length (cm)")
    validator.expect_column_values_to_not_be_null(column="sepal width (cm)")
    validator.expect_column_values_to_be_between(
        column="sepal width (cm)", min_value=2, max_value=4.4
    )
    if fail:
        validator.expect_column_values_to_be_between(
            column="sepal length (cm)", min_value=0, max_value=5
        )

    validator.save_expectation_suite(discard_failed_expectations=False)


def cleanup_expectations():
    dirpath = Path(ROOT_DIRECTORY)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)


def run_expectations():
    fn = mlrun.import_function("function.yaml")
    run = fn.run(
        inputs={"data": "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"},
        params={
            "expectation_suite_name": EXPECTATION_SUITE_NAME,
            "data_asset_name": DATA_ASSET_NAME,
            "root_directory": ROOT_DIRECTORY,
            "datasource_name": DATASOURCE_NAME,
            "data_connector_name": DATA_CONNECTOR_NAME,
        },
        local=True,
    )
    return run


def test_validate_expectations_pass():
    # Setup
    set_expectations(fail=False)
    run = run_expectations()

    # Check that great expectations directory structure was successfully created
    dirpath = Path(ROOT_DIRECTORY)
    assert dirpath.exists()
    assert dirpath.is_dir()

    # Check that run outptuts were successfully saved
    assert "validated" in run.outputs
    assert "validation_results" in run.outputs

    # Check that validation passed
    assert run.outputs["validated"] == True

    # Assert that data docs were saved in run
    assert run.outputs["validation_results"].endswith(".html")

    # Assert that data docs exist on filesystem
    dirpath = Path(run.outputs["validation_results"])
    assert dirpath.exists()

    # Tear down
    cleanup_expectations()

def test_validate_expectations_fail():
    # Setup
    set_expectations(fail=True)
    run = run_expectations()

    # Check that great expectations directory structure was successfully created
    dirpath = Path(ROOT_DIRECTORY)
    assert dirpath.exists()
    assert dirpath.is_dir()

    # Check that run outptuts were successfully saved
    assert "validated" in run.outputs
    assert "validation_results" in run.outputs

    # Check that validation passed
    assert run.outputs["validated"] == False

    # Assert that data docs were saved in run
    assert run.outputs["validation_results"].endswith(".html")

    # Assert that data docs exist on filesystem
    dirpath = Path(run.outputs["validation_results"])
    assert dirpath.exists()

    # Tear down
    cleanup_expectations()