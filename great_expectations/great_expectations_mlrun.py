import os
import shutil

import mlrun

from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult


def get_default_datasource_config(
    datasource_name: str, data_connector_name: str
) -> dict:
    """
    Convenience function to get the default pandas datasource config
    for use in validating expectations.

    :param datasource_name:     Name of datasource.
    :param data_connector_name: Name of data connector.

    :returns: Configuration for default datasource.
    """
    default_datasource_config = {
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
    return default_datasource_config


def get_default_checkpoint_config(checkpoint_name: str) -> dict:
    """
    Convenience function to get the default checkpoint config for
    use in validating expectations.

    :param checkpoint_name: Name of checkpoint.

    :returns: Configuration for default checkpoint.
    """
    return {
        "name": checkpoint_name,
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S-my-run-name-template",
    }


def get_data_doc_path(checkpoint_result: CheckpointResult) -> str:
    """
    Convenience function to get the path of the output
    data doc from a checkpoint result.

    :param checkpoint_result: Great Expectations checkpoint result.

    :returns: Absolute path to new data doc.
    """
    result_id = checkpoint_result.list_validation_result_identifiers()[0]
    data_doc_path = checkpoint_result["run_results"][result_id]["actions_results"][
        "update_data_docs"
    ]["local_site"]
    data_doc_path = data_doc_path.replace("file://", "")
    return data_doc_path


def validate_expectations(
    context: mlrun.MLClientCtx,
    data: mlrun.DataItem,
    expectation_suite_name: str,
    data_asset_name: str,
    datasource_name: str = "pandas_datasource",
    data_connector_name: str = "default_runtime_data_connector_name",
    datasource_config: dict = None,
    batch_identifiers: dict = None,
    root_directory: str = None,
    checkpoint_name: str = None,
    checkpoint_config: dict = None,
) -> None:
    """
    Main function to validate an input dataset, datasource, data connector,
    and expectation suite.

    Runs the Great Expectation validation and logs
    whether the validation was a success as well as the output page
    of the data docs.

    :param context:                MLRun context.
    :param data:                   Data to validate. Can be local or remote link.
    :param expectation_suite_name: Name of expectation suite to validate against.
    :param data_asset_name:        Name of dataset in Great Expectations.
    :param datasource_name:        Name of datasource to use for validation.
    :param data_connector_name:    Name of data connector to use for validation.
    :param datasource_config:      Full configuration for datasource. For use with custom
                                   data sources other than the default pandas datasource.
    :param batch_identifiers:      Custom metadata for identifying particular batches of
                                   data. For use when not using the default batch identifiers.
    :param root_directory:         Path to underlying Great Expectations project. Defaults to
                                   MLRun project artifact path if not specified.
    :param checkpoint_name:        Name of checkpoint to use for validation.
    :param checkpoint_config:      Full configuration for checkpoint. For use with custome
                                   checkpoint config other than the default.
    """

    # Get data
    df = data.as_df()

    # Use default root directory for project if not specified
    root_directory = (
        root_directory
        if root_directory
        else f"/v3io/projects/{context.project}/great_expectations"
    )

    # Load great expectations context
    ge_context = BaseDataContext(
        project_config=DataContextConfig(
            store_backend_defaults=FilesystemStoreBackendDefaults(
                root_directory=root_directory
            )
        )
    )

    # Get expectation suite
    ge_context.get_expectation_suite(expectation_suite_name=expectation_suite_name)

    # Add default data source if not specified
    datasource_config = (
        datasource_config
        if datasource_config
        else get_default_datasource_config(datasource_name, data_connector_name)
    )
    ge_context.add_datasource(**datasource_config)

    # Get data batch
    batch_identifiers = (
        batch_identifiers
        if batch_identifiers
        else {"default_identifier_name": "default_identifier"}
    )
    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name=data_connector_name,
        data_asset_name=data_asset_name,
        runtime_parameters={"batch_data": df},
        batch_identifiers=batch_identifiers,
    )

    # Get validator
    validator = ge_context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name,
    )

    # Use default checkpoint name and config if not specified
    checkpoint_name = (
        checkpoint_name if checkpoint_name else f"{data_asset_name}_checkpoint"
    )
    checkpoint_config = (
        checkpoint_config
        if checkpoint_config
        else get_default_checkpoint_config(checkpoint_name)
    )

    # Add checkpoint
    ge_context.add_checkpoint(**checkpoint_config)

    # Run expectation suite on checkpoint
    checkpoint_result = ge_context.run_checkpoint(
        checkpoint_name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name,
            }
        ],
    )

    # Log success
    context.log_result("validated", checkpoint_result["success"])

    # Log data doc
    data_doc_path = get_data_doc_path(checkpoint_result)
    context.log_artifact("validation_results", target_path=data_doc_path)
