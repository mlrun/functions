# Copyright 2025 Iguazio
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
#This module acts as a lightweight gateway to OpenAI-compatible APIs.
#You can send chat prompts, create embeddings, or get model responses without worrying about authentication or endpoint differences. 
#It simplifies access so you can test, analyze, or integrate AI features directly into your projects or notebooks with minimal setup.


from typing import Dict, Optional, List

class VLLMModule:
    """
    VLLMModule
    
    This module provides a lightweight wrapper for deploying a vLLM
    (OpenAI-compatible) large language model server as an MLRun application runtime.
    
    The VLLMModule is responsible for:
    - Creating an MLRun application runtime based on a vLLM container image
    - Configuring GPU resources, memory limits, and Kubernetes node selection
    - Launching the model using `vllm serve` with configurable runtime flags
    - Supporting multi-GPU inference via tensor parallelism
    - Automatically configuring shared memory (/dev/shm) when using multiple GPUs
    - Exposing an OpenAI-compatible API (e.g. /v1/chat/completions) for inference
    - Providing a simple Python interface for deployment and invocation from Jupyter notebooks
    
    The module is designed to be used in Jupyter notebooks and MLRun pipelines,
    allowing users to deploy and test large language models on Kubernetes
    with minimal configuration.
    """

    def __init__(
            self,
            project: str,
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

        if tensor_parallel_size is not None:
            if tensor_parallel_size < 1:
                raise ValueError("tensor_parallel_size must be >= 1")
            if tensor_parallel_size > gpus:
                raise ValueError(
                    f"tensor_parallel_size ({tensor_parallel_size}) cannot be greater than gpus ({gpus})"
                )

        
        
        if node_selector is None:
            node_selector = {"alpha.eksctl.io/nodegroup-name": "added-gpu"}
        
        if not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")

        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

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

