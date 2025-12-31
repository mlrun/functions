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

import os

import mlrun.errors
from mlrun import code_to_function, get_current_project, mlconf
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaStream,
    DatastoreProfileTDEngine,
    DatastoreProfileV3io,
)
from mlrun.runtimes import ServingRuntime
from mlrun.serving import ModelRunnerStep
from mlrun.utils import logger


class AgentDeployer:
    def __init__(
        self,
        agent_name: str,
        model_class_name: str,
        function: str,
        result_path: str | None = None,
        inputs_path: str | None = None,
        outputs: list[str] | None = None,
        requirements: list[str] | None = None,
        image: str = "mlrun/mlrun",
        set_model_monitoring: bool = False,
        **model_params,
    ):
        """
        Class to deploy an agent as a serving function in MLRun.

        :param agent_name: Name of the agent
        :param model_class_name: Model class name. If LLModel is chosen
                                    (either by name `LLModel` or by its full path, e.g. mlrun.serving.states.LLModel),
                                    outputs will be overridden with UsageResponseKeys fields.
        :param function: Path to the function file.
        :param result_path: when specified selects the key/path in the output event to use as model monitoring
                                      outputs this require that the output event body will behave like a dict,
                                      expects scopes to be defined by dot notation (e.g "data.d").
        :param inputs_path: when specified selects the key/path in the event to use as model monitoring inputs
                                      this require that the event body will behave like a dict, expects scopes to be
                                      defined by dot notation (e.g "data.d").
        :param outputs: list of the model outputs (e.g. labels) ,if provided will override the outputs
                                      that been configured in the model artifact, please note that those outputs need to
                                      be equal to the model_class predict method outputs (length, and order).
        :param requirements: List of additional requirements for the function
        :param image: Docker image to be used for the function
        :param set_model_monitoring: Whether to configure model monitoring
        :param model_params: Parameters for model instantiation
        """

        self._function = None
        self._project = None
        self._project_name = None
        self.agent_name = agent_name
        self.model_class_name = model_class_name
        self.function_file = function
        self.requirements = requirements or []
        self.model_params = model_params or {}
        self.result_path = result_path
        self.inputs_path = inputs_path
        self.output_schema = outputs
        self.image = image
        if set_model_monitoring:
            self.configure_model_monitoring()

    def configure_model_monitoring(self):
        """Configure model monitoring for the active project."""
        if not self.project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "No active project detected, unable to set model monitoring"
            )
        if mlconf.is_ce_mode():
            mlrun_namespace = os.environ.get("MLRUN_NAMESPACE", "mlrun")
            tsdb_profile = DatastoreProfileTDEngine(
                name="tdengine-tsdb-profile",
                user="root",
                password="taosdata",
                host=f"tdengine-tsdb.{mlrun_namespace}.svc.cluster.local",
                port="6041",
            )

            stream_profile = DatastoreProfileKafkaStream(
                name="kafka-stream-profile",
                brokers=f"kafka-stream.{mlrun_namespace}.svc.cluster.local:9092",
                topics=[],
            )
        else:
            tsdb_profile = DatastoreProfileV3io(
                name="v3io-tsdb-profile",
                v3io_access_key=mlconf.get_v3io_access_key(),
            )
            stream_profile = DatastoreProfileV3io(
                name="v3io-stream-profile",
                v3io_access_key=mlconf.get_v3io_access_key(),
            )

        self.project.register_datastore_profile(tsdb_profile)
        self.project.register_datastore_profile(stream_profile)

        self.project.set_model_monitoring_credentials(
            stream_profile_name=stream_profile.name,
            tsdb_profile_name=tsdb_profile.name,
            replace_creds=True,
        )
        try:
            self.project.enable_model_monitoring(
                base_period=10, deploy_histogram_data_drift_app=False
            )
        except (mlrun.errors.MLRunConflictError, mlrun.errors.MLRunHTTPError) as e:
            logger.info(
                "While calling enable_model_monitoring, caught expected exception:",
                error=str(e),
            )

    @property
    def project(self):
        """Get the current MLRun project."""
        if self._project:
            return self._project
        self._project = get_current_project(silent=True)
        return self._project

    @property
    def project_name(self):
        """Get the name of the current MLRun project."""
        if self._project_name:
            return self._project_name
        if self.project:
            self._project_name = self.project.metadata.name
            return self._project_name
        raise mlrun.errors.MLRunInvalidArgumentError(
            "No current project found to get project name"
        )

    def get_function(self) -> ServingRuntime:
        """
        Get the serving function, loading it if necessary.
        """
        if self._function is None:
            self._load_function()
        return self._function

    def deploy_function(self, enable_tracking: bool) -> ServingRuntime:
        """
        Deploy the agent as a serving function in MLRun.
        :param enable_tracking: Whether to enable tracking for the function.
        """

        function = self.get_function()
        function.set_tracking(enable_tracking=enable_tracking)
        function.deploy()
        return function

    def _load_function(
        self,
    ) -> ServingRuntime:
        self._function = code_to_function(
            name=f"{self.agent_name}_serving_function",
            filename=self.function_file,
            project=self.project_name,
            kind="serving",
            image=self.image,
            requirements=self.requirements,
        )
        graph = self._function.set_topology(topology="flow", engine="async")
        model_runner_step = ModelRunnerStep()
        model_runner_step.add_model(
            model_class=self.model_class_name,
            endpoint_name=self.agent_name,
            result_path=self.result_path,
            input_path=self.inputs_path,
            outputs=self.output_schema,
            execution_mechanism="naive",
            **self.model_params,
        )
        graph.to(model_runner_step).respond()
        return self._function
