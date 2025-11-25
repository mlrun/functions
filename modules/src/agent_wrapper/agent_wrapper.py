# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

from mlrun import get_or_create_project, code_to_function, mlconf, function_to_module
from mlrun.serving import ModelRunnerStep, Model, ModelSelector
from mlrun.datastore.datastore_profile import DatastoreProfileV3io

class ModuleModelWrapper(Model):
    def __init__(
            self,
            hub_module_name:str,
            file_name: str,
            model_obj_name: str,
            invoke_methods: list[str],
            **kwargs
    ) -> None:
        super().__init__(name=hub_module_name, raise_exception=True, **kwargs)
        self.model = None
        self.hub_module_name = hub_module_name
        self.file_name = file_name
        self.model_obj_name = model_obj_name
        self.invoke_methods = invoke_methods
        self.load()
        # TODO: add model prompt artifact? do we log the model? do we structure the output? do we serialize it?

    def predict(self, body: Any, **kwargs) -> Any:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() before predict().")
        body["module_outputs"] = {}
        for method_name in self.invoke_methods:
            method = getattr(self.model, method_name)
            if callable(method):
                body["module_outputs"][method_name] = method(body, **kwargs)

    async def predict_async(self, body: Any, **kwargs) -> Any:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() before predict().")
        body["module_outputs"] = {}
        for method_name in self.invoke_methods:
            method = getattr(self.model, method_name)
            if callable(method):
                result = method(body, **kwargs)
                if hasattr(result, "__await__"):
                    result = await result
                body["module_outputs"][method_name] = result
        return body

    def load(self):
        if not self.model:
            mod = function_to_module(self.file_name)
            self.model = getattr(mod, self.model_obj_name)
        return self.model


class AgentDeployer:
    def __init__(
            self,
            project_name: str,
            agent_name: str,
            function: str,
            class_name: str,
            invoke_methods: list[str],
            requirements: list[str] = None,
            image: str = "",
            prompt: str = ""
    ):
        self._project = None
        self.project_name = project_name
        self.agent_name = agent_name
        self.function = function
        self.class_name = class_name
        self.requirements = requirements or []
        self.prompt = prompt
        self.invoke_methods = invoke_methods
        self.image = image or "mlrun/mlrun"
        self.configure_model_monitoring()

    def configure_model_monitoring(self):
        tsdb_profile = DatastoreProfileV3io(
            name="v3io-tsdb-profile",
            v3io_access_key=mlconf.get_v3io_access_key(),
        )
        self.project.register_datastore_profile(tsdb_profile)
        self.project.enable_model_monitoring(base_period=10, image=self.image)

    @property
    def project(self):
        if self._project:
            return self._project
        self._project = get_or_create_project(self.project_name, context="./")
        return self._project

    def get_function(self, enable_tracking: bool = True):
        function = code_to_function(
            name=self.agent_name,
            filename=self.function,
            project=self.project_name,
            kind="serving",
            image=self.image,
            requirements=self.requirements,
        )
        graph = function.set_topology(topology="flow", engine="async")
        model_runner_step = ModelRunnerStep()
        module_wrapper = ModuleModelWrapper(self.agent_name, self.function, self.class_name, self.invoke_methods)
        model_runner_step.add_model(
            model_class=module_wrapper,
            endpoint_name="my-second-model",
            result_path="module_outputs",
            outputs=self.invoke_methods,
            execution_mechanism="naive"
        )
        graph.to(model_runner_step)
        function.set_tracking(enable_tracking=enable_tracking)
        return function
