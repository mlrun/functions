from mlrun import code_to_function
from mlrun.runtimes import RemoteRuntime


def build_stream():
    fn: RemoteRuntime = code_to_function(
        name="model-monitoring-stream",
        kind="nuclio",
        image="mlrun/mlrun",
        filename="stream.py",
        handler="handler",
    )
    fn.spec.no_cache = True
    fn.spec.build.commands = [
        "pip uninstall mlrun --yes",
        "pip install git+https://github.com/mlrun/mlrun.git@development",
        "pip uninstall storey --yes",
        "pip install git+https://github.com/mlrun/storey.git@development",
    ]
    fn.export("stream.yaml")


def build_batch():
    fn: RemoteRuntime = code_to_function(
        name="model-monitoring-batch",
        kind="job",
        image="mlrun/mlrun:0.6.2-rc4",
        filename="/batch.py",
        handler="handler",
    )
    fn.export("batch.yaml")


if __name__ == "__main__":
    build_stream()
    build_batch()
