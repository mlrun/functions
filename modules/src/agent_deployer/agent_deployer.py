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
from typing import Optional
import os

import mlrun.errors
from mlrun import get_current_project, code_to_function, mlconf
from mlrun.serving import ModelRunnerStep
from mlrun.datastore.datastore_profile import DatastoreProfileV3io,DatastoreProfileKafkaStream, DatastoreProfileTDEngine


class AgentDeployer:
    """Class to deploy an agent as a serving function in MLRun."""
    def __init__(
            self,
            agent_name: str,
            model_class_name: str,
            function: str,
            result_path: Optional[str] = None,
            inputs_path: Optional[str] = None,
            output_schema: Optional[list[str]] = None,
            requirements: Optional[list[str]] = None,
            image: Optional[str] = "mlrun/mlrun",
            set_model_monitoring: Optional[bool] = False,
            **model_params

    ):
        self._project = None
        self._project_name = None
        self.agent_name = agent_name
        self.model_class_name = model_class_name
        self.function = function
        self.requirements = requirements or []
        self.model_params = model_params or {}
        self.result_path = result_path
        self.inputs_path = inputs_path
        self.output_schema = output_schema
        self.image = image
        if set_model_monitoring:
            self.configure_model_monitoring()

    def configure_model_monitoring(self):
        """Configure model monitoring for the current project."""
        if not self.project:
            raise mlrun.errors.MLRunInvalidArgumentError("No current project found to set model monitoring")
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
                base_period=10,
                image=self.image,
                deploy_histogram_data_drift_app=False
            )
        except (mlrun.errors.MLRunConflictError, mlrun.errors.MLRunHTTPError) as e:
            print(e)
            pass

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
        raise mlrun.errors.MLRunInvalidArgumentError("No current project found to get project name")

    def deploy_function(self, enable_tracking: bool = True):
        """
        Deploy the agent as a serving function in MLRun.
        :param enable_tracking: Whether to enable tracking for the function.
        """
        function = code_to_function(
            name=f"{self.agent_name}_serving_function",
            filename=self.function,
            project=self.project_name,
            kind="serving",
            image=self.image,
            requirements=self.requirements,
        )
        graph = function.set_topology(topology="flow", engine="async")
        model_runner_step = ModelRunnerStep()
        model_runner_step.add_model(
            model_class=self.model_class_name,
            endpoint_name=self.agent_name,
            result_path=self.result_path,
            input_path=self.inputs_path,
            outputs=self.output_schema,
            execution_mechanism="naive",
            **self.model_params
        )
        graph.to(model_runner_step).respond()
        function.set_tracking(enable_tracking=enable_tracking)
        function.deploy()
        return function
