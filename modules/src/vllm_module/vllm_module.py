from typing import Dict, Optional, List


class VLLMModule:
    def __init__(
            self,
            project,
            *,
            node_selector: Optional[Dict[str, str]] = None,
            name: str = "vllm",
            image: str = "vllm/vllm-openai:latest",
            model: str = "Qwen/Qwen2.5-Omni-3B",
            gpus: int = 1,
            mem: str = "10G",
            port: int = 8000,
            dtype: str = "auto",
            tensor_parallel_size: Optional[int] = None,
            uvicorn_log_level: str = "info",
            max_tokens: int = 500,
    ):
        if gpus < 1:
            raise ValueError("gpus must be >= 1")
        if tensor_parallel_size is not None and tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        if node_selector is None:
            node_selector = {"alpha.eksctl.io/nodegroup-name": "added-gpu"}

        self.project = project
        self.name = name
        self.image = image
        self.model = model
        self.gpus = gpus
        self.mem = mem
        self.node_selector = node_selector
        self.port = port
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.uvicorn_log_level = uvicorn_log_level
        self.max_tokens = max_tokens

        self.vllm_app = self.project.set_function(
            name=self.name,
            kind="application",
            image=self.image,
        )

        self.vllm_app.with_limits(gpus=self.gpus, mem=self.mem)

        if self.node_selector:
            self.vllm_app.with_node_selection(node_selector=self.node_selector)

        self.vllm_app.set_internal_application_port(self.port)

        args: List[str] = [
            "serve",
            self.model,
            "--dtype",
            self.dtype,
            "--port",
            str(self.port),
        ]

        if self.uvicorn_log_level:
            args += ["--uvicorn-log-level", self.uvicorn_log_level]

        if self.gpus > 1:
            tps = self.tensor_parallel_size or self.gpus
            args += ["--tensor-parallel-size", str(tps)]

            # For more than one GPU you should create a share volume for the multiple GPUs
            self.vllm_app.spec.volumes = [{"name": "dshm", "emptyDir": {"medium": "Memory"}}]
            self.vllm_app.spec.volume_mounts = [{"name": "dshm", "mountPath": "/dev/shm"}]

        if max_tokens < 0:
            self.max_tokens = 500

        self.vllm_app.spec.command = "vllm"
        self.vllm_app.spec.args = args

        self.vllm_app.spec.min_replicas = 1
        self.vllm_app.spec.max_replicas = 1

    def get_runtime(self):
        return self.vllm_app

    def add_args(self, extra_args: List[str]):
        if not isinstance(extra_args, list) or not all(isinstance(x, str) for x in extra_args):
            raise ValueError("extra_args must be a list of strings")
        self.vllm_app.spec.args += extra_args

