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

# Choose our model's name:
MODEL_NAME = "model"

# Choose our ONNX version model's name:
ONNX_MODEL_NAME = f"onnx_{MODEL_NAME}"

# Choose our optimized ONNX version model's name:
OPTIMIZED_ONNX_MODEL_NAME = f"optimized_{ONNX_MODEL_NAME}"


def _setup_environment() -> str:
    """
    Setup the test environment, creating the artifacts path of the test.

    :returns: The temporary directory created for the test artifacts path.
    """
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

    # Clean the artifacts directory:
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


def test_to_onnx_help():
    """
    Test the 'to_onnx' handler, passing "help" in the 'framework_kwargs'.
    """
    # Setup the tests environment:
    artifact_path = _setup_environment()

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_run = log_model_function.run(
        handler="_log_tf_keras_model",
        artifact_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml")

    # Run the function, passing "help" in 'framework_kwargs' and see that no exception was raised:
    is_test_passed = True
    try:
        onnx_function.run(
            handler="to_onnx",
            artifact_path=artifact_path,
            params={
                # Take the logged model from the previous function.
                "model_path": log_model_run.status.artifacts[0]["spec"]["target_path"],
                "load_model_kwargs": {"model_name": MODEL_NAME},
                "framework_kwargs": "help",
            },
            local=True,
        )
    except TypeError as exception:
        print(
            f"The test failed, the help was not handled properly and raised the following error: {exception}"
        )
        is_test_passed = False

    # Cleanup the tests environment:
    _cleanup_environment(artifact_path=artifact_path)

    assert is_test_passed


def test_tf_keras_to_onnx():
    """
    Test the 'to_onnx' handler, giving it a tf.keras model.
    """
    # Setup the tests environment:
    artifact_path = _setup_environment()

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_run = log_model_function.run(
        handler="_log_tf_keras_model",
        artifact_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml")

    # Run the function to convert our model to ONNX:
    onnx_function_run = onnx_function.run(
        handler="to_onnx",
        artifact_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": log_model_run.status.artifacts[0]["spec"]["target_path"],
            "load_model_kwargs": {"model_name": MODEL_NAME},
            "onnx_model_name": ONNX_MODEL_NAME,
        },
        local=True,
    )

    # Cleanup the tests environment:
    _cleanup_environment(artifact_path=artifact_path)

    # Print the outputs list:
    print(f"Produced outputs: {onnx_function_run.outputs}")

    # Verify the '.onnx' model was created:
    assert "model" in onnx_function_run.outputs


def test_pytorch_to_onnx():
    """
    Test the 'to_onnx' handler, giving it a pytorch model.
    """
    # Setup the tests environment:
    artifact_path = _setup_environment()

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_run = log_model_function.run(
        handler="_log_pytorch_model",
        artifact_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml")

    # Run the function to convert our model to ONNX:
    onnx_function_run = onnx_function.run(
        handler="to_onnx",
        artifact_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": log_model_run.status.artifacts[1]["spec"]["target_path"],
            "load_model_kwargs": {
                "model_name": MODEL_NAME,
                "model_class": "mobilenet_v2",
                "modules_map": log_model_run.status.artifacts[0]["spec"]["target_path"],
            },
            "onnx_model_name": ONNX_MODEL_NAME,
            "framework_kwargs": {"input_signature": [((32, 3, 224, 224), "float32")]},
        },
        local=True,
    )

    # Cleanup the tests environment:
    _cleanup_environment(artifact_path=artifact_path)

    # Print the outputs list:
    print(f"Produced outputs: {onnx_function_run.outputs}")

    # Verify the '.onnx' model was created:
    assert "model" in onnx_function_run.outputs


def test_optimize_help():
    """
    Test the 'optimize' handler, passing "help" in the 'optimizations'.
    """
    # Setup the tests environment:
    artifact_path = _setup_environment()

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml")

    # Run the function, passing "help" in 'optimizations' and see that no exception was raised:
    is_test_passed = True
    try:
        onnx_function.run(
            handler="optimize",
            artifact_path=artifact_path,
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

    # Cleanup the tests environment:
    _cleanup_environment(artifact_path=artifact_path)

    assert is_test_passed


def test_optimize():
    """
    Test the 'optimize' handler, giving it a model from the ONNX zoo git repository.
    """
    # Setup the tests environment:
    artifact_path = _setup_environment()

    # Create the function:
    log_model_function = mlrun.code_to_function(
        filename="test_onnx_utils.py",
        name="log_model",
        kind="job",
        image="mlrun/ml-models",
    )

    # Run the function to log the model:
    log_model_run = log_model_function.run(
        handler="_log_tf_keras_model",
        artifact_path=artifact_path,
        params={"model_name": MODEL_NAME},
        local=True,
    )

    # Import the ONNX Utils function:
    onnx_function = mlrun.import_function("function.yaml")

    # Run the function to convert our model to ONNX:
    to_onnx_function_run = onnx_function.run(
        handler="to_onnx",
        artifact_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": log_model_run.status.artifacts[0]["spec"]["target_path"],
            "load_model_kwargs": {"model_name": MODEL_NAME},
            "onnx_model_name": ONNX_MODEL_NAME,
        },
        local=True,
    )

    # Run the function to optimize our model:
    optimize_function_run = onnx_function.run(
        handler="optimize",
        artifact_path=artifact_path,
        params={
            # Take the logged model from the previous function.
            "model_path": to_onnx_function_run.status.artifacts[0]["spec"][
                "target_path"
            ],
            "handler_init_kwargs": {"model_name": ONNX_MODEL_NAME},
            "optimized_model_name": OPTIMIZED_ONNX_MODEL_NAME,
        },
        local=True,
    )

    # Cleanup the tests environment:
    _cleanup_environment(artifact_path=artifact_path)

    # Print the outputs list:
    print(f"Produced outputs: {optimize_function_run.outputs}")

    # Verify the '.onnx' model was created:
    assert "model" in optimize_function_run.outputs
