from mlrun import code_to_function
from mlrun.runtimes import RemoteRuntime


fn: RemoteRuntime = code_to_function(
    name="model-monitoring-stream",
    kind="nuclio",
    image="mlrun/mlrun",
    filename="model_monitoring_stream.py",
    handler="handler",
)
fn.export("model_monitoring_stream.yaml")
