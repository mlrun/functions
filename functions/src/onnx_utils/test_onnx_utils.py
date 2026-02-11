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
import shutil
import tempfile

import mlrun
import pytest

PROJECT_NAME = "onnx-utils"

# Choose our model's name:
MODEL_NAME = "model"

# Choose our ONNX version model's name:
ONNX_MODEL_NAME = f"onnx_{MODEL_NAME}"

# Choose our optimized ONNX version model's name:
OPTIMIZED_ONNX_MODEL_NAME = f"optimized_{ONNX_MODEL_NAME}"

REQUIRED_ENV_VARS = [
    "MLRUN_DBPATH",
    "MLRUN_ARTIFACT_PATH",
    "V3IO_USERNAME",
    "V3IO_ACCESS_KEY",
]


def _validate_environment_variables() -> bool:
    """
    Checks that all required Environment variables are set.
    """
    environment_keys = os.environ.keys()
    return all(key in environment_keys for key in REQUIRED_ENV_VARS)


def _is_tf2onnx_available() -> bool:
    """
    Check if tf2onnx is installed (required for TensorFlow/Keras ONNX conversion).
    """
    try:
        import tf2onnx
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def onnx_project():
    """Create/get the MLRun project once per test session."""
    return mlrun.get_or_create_project(PROJECT_NAME, context="./")


@pytest.fixture(autouse=True)
def test_environment(onnx_project):
    """Setup and cleanup test artifacts for each test."""
    artifact_path = tempfile.mkdtemp()
    yield artifact_path
    # Cleanup - only remove files/dirs from the directory containing this test file,
    # never from an arbitrary CWD (which could be the project root).
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for test_output in [
        "schedules",
        "runs",
        "artifacts",
        "functions",
        "model.pt",
        "model.zip",
        "model_modules_map.json",
        "model_modules_map.json.json",
        "onnx_model.onnx",
        "optimized_onnx_model.onnx",
    ]:
        test_output_path = os.path.join(test_dir, test_output)
        if os.path.exists(test_output_path):
            if os.path.isdir(test_output_path):
                shutil.rmtree(test_output_path)
            else:
                os.remove(test_output_path)
    if os.path.exists(artifact_path):
        shutil.rmtree(artifact_path)


def _log_tf_keras_model(context: mlrun.MLClientCtx, model_name: str):
    """
    Create and log a tf.keras model - MobileNetV2.

    :param context:    The context to log to.
    :param model_name: The model name to use.
    """
    # To use `tf_keras` instead of `tensorflow.keras`
    os.environ["TF_USE_LEGACY_KERAS"] = "true"
    from mlrun.frameworks.tf_keras import TFKerasModelHandler
    from tensorflow import keras

    # Download the MobileNetV2 model:
    model = keras.applications.mobilenet_v2.MobileNetV2()

    # Initialize a model handler for logging the model:
    model_handler = TFKerasModelHandler(
        model_name=model_name, model=model, context=context
    )

    # Log the model:
    model_handler.log()


def _log_pytorch_model(context: mlrun.MLClientCtx, model_name: str):
    """
    Create and log a pytorch model - MobileNetV2.

    :param context:    The context to log to.
    :param model_name: The model name to use.
    """
    import torchvision
    from mlrun.frameworks.pytorch import PyTorchModelHandler

    # Download the MobileNetV2 model:
    model = torchvision.models.mobilenet_v2()

    # Initialize a model handler for logging the model:
    model_handler = PyTorchModelHandler(
        model_name=model_name,
        model=model,
        model_class="mobilenet_v2",
        modules_map={"torchvision.models": "mobilenet_v2"},
        context=context,
    )

    # Log the model:
    model_handler.log()


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_to_onnx_help(test_environment):
    """
    Test the 'to_onnx' handler, passing "help" in the 'framework_kwargs'.
    """
    artifact_path = test_environment

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        project=PROJECT_NAME,
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_function.run(
        handler="_log_pytorch_model",
        output_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Get artifact paths - construct from artifact_path and run structure
    run_artifact_dir = os.path.join(artifact_path, "log-model--log-pytorch-model", "0")
    model_path = os.path.join(run_artifact_dir, "model")
    modules_map_path = os.path.join(run_artifact_dir, "model_modules_map.json.json")

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml", project=PROJECT_NAME)

    # Run the function, passing "help" in 'framework_kwargs' and see that no exception was raised:
    is_test_passed = True
    try:
        onnx_function.run(
            handler="to_onnx",
            output_path=artifact_path,
            params={
                # Take the logged model from the previous function.
                "model_path": model_path,
                "load_model_kwargs": {
                    "model_name": MODEL_NAME,
                    "model_class": "mobilenet_v2",
                    "modules_map": modules_map_path,
                },
                "framework_kwargs": "help",
            },
            local=True,
        )
    except TypeError as exception:
        print(
            f"The test failed, the help was not handled properly and raised the following error: {exception}"
        )
        is_test_passed = False

    assert is_test_passed


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
@pytest.mark.skipif(
    condition=not _is_tf2onnx_available(),
    reason="tf2onnx is not installed",
)
def test_tf_keras_to_onnx(test_environment):
    """
    Test the 'to_onnx' handler, giving it a tf.keras model.
    """
    artifact_path = test_environment

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        project=PROJECT_NAME,
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_run = log_model_function.run(
        handler="_log_tf_keras_model",
        output_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml", project=PROJECT_NAME)

    # Run the function to convert our model to ONNX:
    onnx_function_run = onnx_function.run(
        handler="to_onnx",
        output_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": log_model_run.status.artifacts[0]["spec"]["target_path"],
            "load_model_kwargs": {"model_name": MODEL_NAME},
            "onnx_model_name": ONNX_MODEL_NAME,
        },
        local=True,
    )

    # Print the outputs list:
    print(f"Produced outputs: {onnx_function_run.outputs}")

    # Verify the '.onnx' model was created:
    assert "model" in onnx_function_run.outputs


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_pytorch_to_onnx(test_environment):
    """
    Test the 'to_onnx' handler, giving it a pytorch model.
    """
    artifact_path = test_environment

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        project=PROJECT_NAME,
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_run = log_model_function.run(
        handler="_log_pytorch_model",
        output_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml", project=PROJECT_NAME)

    # Get artifact paths - construct from artifact_path and run structure
    run_artifact_dir = os.path.join(artifact_path, "log-model--log-pytorch-model", "0")
    model_path = os.path.join(run_artifact_dir, "model")
    modules_map_path = os.path.join(run_artifact_dir, "model_modules_map.json.json")

    # Run the function to convert our model to ONNX:
    onnx_function_run = onnx_function.run(
        handler="to_onnx",
        output_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": model_path,
            "load_model_kwargs": {
                "model_name": MODEL_NAME,
                "model_class": "mobilenet_v2",
                "modules_map": modules_map_path,
            },
            "onnx_model_name": ONNX_MODEL_NAME,
            "framework_kwargs": {"input_signature": [((32, 3, 224, 224), "float32")]},
        },
        local=True,
    )

    # Print the outputs list:
    print(f"Produced outputs: {onnx_function_run.outputs}")

    # Verify the '.onnx' model was created:
    assert "model" in onnx_function_run.outputs


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_optimize_help(test_environment):
    """
    Test the 'optimize' handler, passing "help" in the 'optimizations'.
    """
    artifact_path = test_environment

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml", project=PROJECT_NAME)

    # Run the function, passing "help" in 'optimizations' and see that no exception was raised:
    is_test_passed = True
    try:
        onnx_function.run(
            handler="optimize",
            output_path=artifact_path,
            params={
                "model_path": "",
                "optimizations": "help",
            },
            local=True,
        )
    except TypeError as exception:
        print(
            f"The test failed, the help was not handled properly and raised the following error: {exception}"
        )
        is_test_passed = False

    assert is_test_passed


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_optimize(test_environment):
    """
    Test the 'optimize' handler, giving it a pytorch model converted to ONNX.
    """
    artifact_path = test_environment

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        project=PROJECT_NAME,
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_function.run(
        handler="_log_pytorch_model",
        output_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Get artifact paths - construct from artifact_path and run structure
    run_artifact_dir = os.path.join(artifact_path, "log-model--log-pytorch-model", "0")
    model_path = os.path.join(run_artifact_dir, "model")
    modules_map_path = os.path.join(run_artifact_dir, "model_modules_map.json.json")

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml", project=PROJECT_NAME)

    # Run the function to convert our model to ONNX:
    onnx_function.run(
        handler="to_onnx",
        output_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": model_path,
            "load_model_kwargs": {
                "model_name": MODEL_NAME,
                "model_class": "mobilenet_v2",
                "modules_map": modules_map_path,
            },
            "onnx_model_name": ONNX_MODEL_NAME,
            "framework_kwargs": {"input_signature": [((32, 3, 224, 224), "float32")]},
        },
        local=True,
    )

    # Get the ONNX model path from the to_onnx run output
    onnx_run_artifact_dir = os.path.join(
        artifact_path, "onnx-utils-to-onnx", "0"
    )
    onnx_model_path = os.path.join(onnx_run_artifact_dir, "model")

    # Run the function to optimize our model:
    optimize_function_run = onnx_function.run(
        handler="optimize",
        output_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": onnx_model_path,
            "handler_init_kwargs": {"model_name": ONNX_MODEL_NAME},
            "optimized_model_name": OPTIMIZED_ONNX_MODEL_NAME,
        },
        local=True,
    )

    # Print the outputs list:
    print(f"Produced outputs: {optimize_function_run.outputs}")

    # Verify the '.onnx' model was created:
    assert "model" in optimize_function_run.outputs
