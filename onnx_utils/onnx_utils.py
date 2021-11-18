from typing import Any, Callable, Dict, List, Tuple

import mlrun
import numpy as np
from mlrun.datastore import store_manager


def _get_framework(model_path: str) -> str:
    """
    Get the framework of the model stored in the given path.

    :param model_path: The model path store object.

    :returns: The model's framework.
    """
    model_artifact, _ = store_manager.get_store_artifact(model_path)
    return model_artifact.labels["framework"]


class _ToONNXConversions:
    """
    An ONNX conversion functions library class.
    """

    @staticmethod
    def tf_keras_to_onnx(
        context: mlrun.MLClientCtx,
        model_name: str,
        model_path: str,
        onnx_model_name: str = None,
        optimize_model: bool = True,
        input_signature: List[Tuple[Tuple[int], str]] = None,
    ):
        """
        Convert a tf.keras model to an ONNX model and log it back to MLRun as a new model object.

        :param context:         The MLRun function execution context
        :param model_name:      The model's name.
        :param model_path:      The model path store object.
        :param onnx_model_name: The name to use to log the converted ONNX model. If not given, the given `model_name`
                                will be used with an additional suffix `_onnx`. Defaulted to None.
        :param optimize_model:  Whether or not to optimize the ONNX model using 'onnxoptimizer' before saving the model.
                                Defaulted to True.
        :param input_signature: A list of the input layers shape and data type properties. Expected to receive a list
                                where each element is an input layer tuple. An input layer tuple is a tuple of:
                                [0] = Layer's shape, a tuple of integers.
                                [1] = Layer's data type, a dtype numpy string.
                                If None, the input signature will be tried to be read automatically before converting to
                                ONNX. Defaulted to None.
        """
        # Import the handler:
        from mlrun.frameworks.tf_keras import TFKerasModelHandler

        # Initialize the handler:
        model_handler = TFKerasModelHandler(
            model_name=model_name,
            model_path=model_path,
            context=context,
        )

        # Load the tf.keras model:
        model_handler.load()

        # Parse the 'input_signature' parameter:
        if input_signature is not None:
            input_signature = [
                np.zeros(shape=shape, dtype=dtype) for (shape, dtype) in input_signature
            ]

        # Convert to ONNX:
        model_handler.to_onnx(
            model_name=onnx_model_name,
            input_signature=input_signature,
            optimize=optimize_model,
        )


# Map for getting the conversion function according to the provided framework:
_CONVERSION_MAP = {
    "tf.keras": _ToONNXConversions.tf_keras_to_onnx
}  # type: Dict[str, Callable]


def to_onnx(
    context: mlrun.MLClientCtx,
    model_name: str,
    model_path: str,
    onnx_model_name: str = None,
    optimize_model: bool = True,
    framework: str = None,
    framework_kwargs: Dict[str, Any] = None,
):
    """
    Convert the given model to an ONNX model.

    :param context:          The MLRun function execution context
    :param model_name:       The model's name.
    :param model_path:       The model path store object.
    :param onnx_model_name:  The name to use to log the converted ONNX model. If not given, the given `model_name` will
                             be used with an additional suffix `_onnx`. Defaulted to None.
    :param optimize_model:   Whether to optimize the ONNX model using 'onnxoptimizer' before saving the model. Defaulted
                             to True.
    :param framework:        The model's framework. If None, it will be read from the 'framework' label of the model
                             artifact provided. Defaulted to None.
    :param framework_kwargs: Additional arguments each framework may require in order to convert to ONNX. To get the doc
                             string of the desired framework onnx conversion function, pass "help".
    """
    # Get the model's framework if needed:
    if framework is None:
        framework = _get_framework(model_path=model_path)

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
            context=context,
            model_name=model_name,
            model_path=model_path,
            onnx_model_name=onnx_model_name,
            optimize_model=optimize_model,
            **framework_kwargs
        )
    except TypeError as exception:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"ERROR: A TypeError exception was raised during the conversion:\n{exception}. "
            f"Please read the {framework} framework conversion function doc string by passing 'help' in the "
            f"'framework_kwargs' dictionary parameter."
        )


def optimize(
    context: mlrun.MLClientCtx,
    model_name: str,
    model_path: str,
    optimizations: List[str] = None,
    fixed_point: bool = False,
    optimized_model_name: str = None,
):
    """
    Optimize the given ONNX model.

    :param context:              The MLRun function execution context.
    :param model_name:           The model's name.
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
        available_passes = '\n* '.join(onnxoptimizer.get_available_passes())
        print(f"The available optimizations are:\n* {available_passes}")
        return

    # Create the model handler:
    model_handler = ONNXModelHandler(
        model_name=model_name, model_path=model_path, context=context
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
