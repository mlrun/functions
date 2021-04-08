from mlrun import code_to_function
from mlrun.runtimes import RemoteRuntime


fn: RemoteRuntime = code_to_function(
    name="model-monitoring-batch",
    kind="job",
    image="mlrun/mlrun",
    filename="model_monitoring_batch.py",
    handler="handler",
)

fn.export("model_monitoring_batch.yaml")
