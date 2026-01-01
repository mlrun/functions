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

import mlrun
from vllm_module import VLLMModule


class TestVllmModule:
    """Test suite for VLLMModule class."""

    def setup_method(self):
        project = mlrun.new_project("vllm", save=False)

        # if your VLLMModule requires node_selector as keyword-only, keep it here
        self.TestVllmModule = VLLMModule(
            project,
            node_selector={"alpha.eksctl.io/nodegroup-name": "added-gpu"},
        )

    def test_vllm_module(self):
        assert (
            type(self.TestVllmModule.vllm_app)
            == mlrun.runtimes.nuclio.application.application.ApplicationRuntime
        )
