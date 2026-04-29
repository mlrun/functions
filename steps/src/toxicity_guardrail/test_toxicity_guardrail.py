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

from toxicity_guardrail import ToxicityGuardrailStep


class TestToxicityGuardrailStep:
    """Test suite for ToxicityGuardrailStep class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        project = mlrun.new_project("toxicity-guardrail", save=False)
        self.fn = project.set_function(
            "toxicity_guardrail.py",
            name="guardrail-fn",
            kind="serving",
            image="mlrun/mlrun",
        )
        graph = self.fn.set_topology("flow", engine="async")
        graph.to(
            class_name="ToxicityGuardrailStep",
            name="toxicity_guardrail",
            threshold=0.5,
        ).respond()

    def test_toxicity_guardrail_step(self):
        """Test that the serving function is correctly configured with ToxicityGuardrailStep."""
        assert type(self.fn) == mlrun.runtimes.ServingRuntime
