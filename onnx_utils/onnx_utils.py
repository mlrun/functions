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
from typing import Any, Callable, Dict, List, Tuple

import mlrun


class _ToONNXConversions:
    """
    An ONNX conversion functions library class.
    """

    @staticmethod
    def tf_keras_to_onnx(
        model_handler,
        onnx_model_name: str = None,
        optimize_model: bool = True,
        input_signature: List[Tuple[Tuple[int], str]] = None,
    ):
        """
        Convert a TF.Keras model to an ONNX model and log it back to MLRun as a new model object.

        :param model_handler:   An initialized TFKerasModelHandler with a loaded model to convert to ONNX.
        :param onnx_model_name: The name to use to log the converted ONNX model. If not given, the given `model_name`
                                will be used with an additional suffix `_onnx`. Defaulted to None.
        :param optimize_model:  Whether or not to optimize the ONNX model using 'onnxoptimizer' before saving the model.
                                Defaulted to True.
        :param input_signature: A list of the input layers shape and data type properties. Expected to receive a list
                                where each element is an input layer tuple. An input layer tuple is a tuple of:
                                [0] = Layer's shape, a tuple of integers.
                                [1] = Layer's data type, a mlrun.data_types.ValueType string.
                                If None, the input signature will be tried to be read from the model artifact. Defaulted
                                to None.
        """
        # Import the framework and handler:
        import tensorflow as tf
        from mlrun.frameworks.tf_keras import TFKerasUtils

        # Check the given 'input_signature' parameter:
        if input_signature is None:
            # Read the inputs from the model:
            try:
                model_handler.read_inputs_from_model()
            except Exception as error:
                raise mlrun.errors.MLRunRuntimeError(
                    f"Please provide the 'input_signature' parameter. The function tried reading the input layers "
                    f"information automatically but failed with the following error: {error}"
                )
        else:
            # Parse the 'input_signature' parameter:
            input_signature = [
                tf.TensorSpec(
                    shape=shape,
                    dtype=TFKerasUtils.convert_value_type_to_tf_dtype(
                        value_type=value_type
                    ),
                )
                for (shape, value_type) in input_signature
            ]

        # Convert to ONNX:
        model_handler.to_onnx(
            model_name=onnx_model_name,
            input_signature=input_signature,
            optimize=optimize_model,
        )

    @staticmethod
    def pytorch_to_onnx(
        model_handler,
        onnx_model_name: str = None,
        optimize_model: bool = True,
        input_signature: List[Tuple[Tuple[int, ...], str]] = None,
        input_layers_names: List[str] = None,
        output_layers_names: List[str] = None,
        dynamic_axes: Dict[str, Dict[int, str]] = None,
        is_batched: bool = True,
    ):
        """
        Convert a PyTorch model to an ONNX model and log it back to MLRun as a new model object.

        :param model_handler:       An initialized PyTorchModelHandler with a loaded model to convert to ONNX.
        :param onnx_model_name:     The name to use to log the converted ONNX model. If not given, the given
                                    `model_name` will be used with an additional suffix `_onnx`. Defaulted to None.
        :param optimize_model:      Whether or not to optimize the ONNX model using 'onnxoptimizer' before saving the
                                    model. Defaulted to True.
        :param input_signature:     A list of the input layers shape and data type properties. Expected to receive a
                                    list where each element is an input layer tuple. An input layer tuple is a tuple of:
                                    [0] = Layer's shape, a tuple of integers.
                                    [1] = Layer's data type, a mlrun.data_types.ValueType string.
                                    If None, the input signature will be tried to be read from the model artifact.
                                    Defaulted to None.
        :param input_layers_names:  List of names to assign to the input nodes of the graph in order. All of the other
                                    parameters (inner layers) can be set as well by passing additional names in the
                                    list. The order is by the order of the parameters in the model. If None, the inputs
                                    will be read from the handler's inputs. If its also None, it is defaulted to:
                                    "input_0", "input_1", ...
        :param output_layers_names: List of names to assign to the output nodes of the graph in order. If None, the
                                    outputs will be read from the handler's outputs. If its also None, it is defaulted
                                    to: "output_0" (for multiple outputs, this parameter must be provided).
        :param dynamic_axes:        If part of the input / output shape is dynamic, like (batch_size, 3, 32, 32) you can
                                    specify it by giving a dynamic axis to the input / output layer by its name as
                                    follows: {
                                        "input layer name": {0: "batch_size"},
                                        "output layer name": {0: "batch_size"},
                                    }
                                    If provided, the 'is_batched' flag will be ignored. Defaulted to None.
        :param is_batched:          Whether to include a batch size as the first axis in every input and output layer.
                                    Defaulted to True. Will be ignored if 'dynamic_axes' is provided.
        """
        # Import the framework and handler:
        import torch
        from mlrun.frameworks.pytorch import PyTorchUtils

        # Parse the 'input_signature' parameter:
        if input_signature is not None:
            input_signature = tuple(
                [
                    torch.zeros(
                        size=shape,
                        dtype=PyTorchUtils.convert_value_type_to_torch_dtype(
                            value_type=value_type
                        ),
                    )
                    for (shape, value_type) in input_signature
                ]
            )

        # Convert to ONNX:
        model_handler.to_onnx(
            model_name=onnx_model_name,
            input_sample=input_signature,
            optimize=optimize_model,
            input_layers_names=input_layers_names,
            output_layers_names=output_layers_names,
            dynamic_axes=dynamic_axes,
            is_batched=is_batched
        )


# Map for getting the conversion function according to the provided framework:
_CONVERSION_MAP = {
    "tensorflow.keras": _ToONNXConversions.tf_keras_to_onnx,
    "torch": _ToONNXConversions.pytorch_to_onnx,
}  # type: Dict[str, Callable]


def to_onnx(
    context: mlrun.MLClientCtx,
    model_path: str,
    onnx_model_name: str = None,
    optimize_model: bool = True,
    framework_kwargs: Dict[str, Any] = None,
):
    """
    Convert the given model to an ONNX model.

    :param context:          The MLRun function execution context
    :param model_path:       The model path store object.
    :param onnx_model_name:  The name to use to log the converted ONNX model. If not given, the given `model_name` will
                             be used with an additional suffix `_onnx`. Defaulted to None.
    :param optimize_model:   Whether to optimize the ONNX model using 'onnxoptimizer' before saving the model. Defaulted
                             to True.
    :param framework_kwargs: Additional arguments each framework may require in order to convert to ONNX. To get the doc
                             string of the desired framework onnx conversion function, pass "help".
    """
    from mlrun.frameworks.auto_mlrun.auto_mlrun import AutoMLRun

    # Get a model handler of the required framework:
    model_handler = AutoMLRun.load_model(model_path=model_path, context=context)

    # Get the model's framework:
    framework = model_handler.FRAMEWORK_NAME

    # Use the conversion map to get the specific framework to onnx conversion:
    if framework not in _CONVERSION_MAP:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The following framework: '{framework}', has no ONNX conversion."
        )
    conversion_function = _CONVERSION_MAP[framework]

    # Check if needed to print the function's doc string ("help" is passed):
    if framework_kwargs == "help":
        print(conversion_function.__doc__)
        return

    # Set the default empty framework kwargs if needed:
    if framework_kwargs is None:
        framework_kwargs = {}

    # Run the conversion:
    try:
        conversion_function(
            model_handler=model_handler,
            onnx_model_name=onnx_model_name,
            optimize_model=optimize_model,
            **framework_kwargs,
        )
    except TypeError as exception:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"ERROR: A TypeError exception was raised during the conversion:\n{exception}. "
            f"Please read the {framework} framework conversion function doc string by passing 'help' in the "
            f"'framework_kwargs' dictionary parameter."
        )


def optimize(
    context: mlrun.MLClientCtx,
    model_path: str,
    optimizations: List[str] = None,
    fixed_point: bool = False,
    optimized_model_name: str = None,
):
    """
    Optimize the given ONNX model.

    :param context:              The MLRun function execution context.
    :param model_path:           Path to the ONNX model object.
    :param optimizations:        List of possible optimizations. To see what optimizations are available, pass "help".
                                 If None, all of the optimizations will be used. Defaulted to None.
    :param fixed_point:          Optimize the weights using fixed point. Defaulted to False.
    :param optimized_model_name: The name of the optimized model. If None, the original model will be overridden.
                                 Defaulted to None.
    """
    # Import the model handler:
    import onnxoptimizer
    from mlrun.frameworks.onnx import ONNXModelHandler

    # Check if needed to print the available optimizations ("help" is passed):
    if optimizations == "help":
        available_passes = "\n* ".join(onnxoptimizer.get_available_passes())
        print(f"The available optimizations are:\n* {available_passes}")
        return

    # Create the model handler:
    model_handler = ONNXModelHandler(
        model_path=model_path, context=context
    )

    # Load the ONNX model:
    model_handler.load()

    # Optimize the model using the given configurations:
    model_handler.optimize(optimizations=optimizations, fixed_point=fixed_point)

    # Rename if needed:
    if optimized_model_name is not None:
        model_handler.set_model_name(model_name=optimized_model_name)

    # Log the optimized model:
    model_handler.log()
