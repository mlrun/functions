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
import shutil
import pytest

import mlrun
from mlrun import import_function

EXPERIMENT_NAME = "azure-automl-test"
PROJECT_NAME = "azure-automl-project"

DATA_URL = "https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv"

SECRETS_REQUIRED_FIELDS = [
    "AZURE_TENANT_ID",
    "AZURE_SERVICE_PRINCIPAL_ID",
    "AZURE_SERVICE_PRINCIPAL_PASSWORD",
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_WORKSPACE_NAME",
    "AZURE_STORAGE_CONNECTION_STRING",
]


def _validate_environment_variables() -> bool:
    environment_keys = os.environ.keys()
    return all(key in environment_keys for key in SECRETS_REQUIRED_FIELDS)


def _get_secrets_spec():
    return mlrun.new_task().with_secrets(
        "env",
        ",".join(SECRETS_REQUIRED_FIELDS),
    )


def _set_environment():
    artifact_path = tempfile.TemporaryDirectory().name
    os.makedirs(artifact_path)
    return artifact_path


def _cleanup_environment(artifact_path: str):
    """
    Cleanup the test environment, deleting files and artifacts created during the test.

    :param artifact_path: The artifact path to delete.
    """
    # Clean the local directory:
    for test_output in [
        *os.listdir(artifact_path),
        "schedules",
        "runs",
        "artifacts",
        "functions",
    ]:
        test_output_path = os.path.abspath(f"./{test_output}")
        if os.path.exists(test_output_path):
            if os.path.isdir(test_output_path):
                shutil.rmtree(test_output_path)
            else:
                os.remove(test_output_path)

    # Clean the artifacts' directory:
    shutil.rmtree(artifact_path)


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="AzureML secrets should be provided as environment variables",
)
def test_train():
    """
    Test the 'train' handler with iris dataset.
    """
    test_pass = False

    # Setting secrets:
    secrets_spec = _get_secrets_spec()

    # Setting environment:
    artifact_path = _set_environment()
    azure_automl_fn = import_function("function.yaml")
    model_paths, save_n_models = [], 2

    try:
        azureml_run = azure_automl_fn.run(
            runspec=secrets_spec,
            handler="train",
            params={
                "experiment_name": EXPERIMENT_NAME,
                "cpu_cluster_name": "azureml-cpu",
                "dataset_name": "iris-test",
                "log_azure": False,
                "dataset_description": "iris training data",
                "label_column_name": "label",
                "create_new_version": True,
                "register_model_name": "iris-model",
                "save_n_models": save_n_models,
            },
            inputs={"dataset": DATA_URL},
            artifact_path=artifact_path,
            local=True,
        )
        # Get trained models:
        num_saved_models = len(azureml_run.status.iterations) - 1  # The first one in the list is the 'columns'
        test_pass = num_saved_models == save_n_models

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")

    _cleanup_environment(artifact_path)

    assert test_pass, f'Created {len(model_paths)} models instead of {save_n_models}'
