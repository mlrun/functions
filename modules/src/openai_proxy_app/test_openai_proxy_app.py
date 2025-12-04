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

from openai_proxy_app import OpenAIModule
import mlrun

class TestOpenAIProxyApp:
    """Test suite for TestOpenAIProxyApp class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        project = mlrun.new_project("openai", save=False)
        self.TestOpenAIProxyApp = OpenAIModule(project)

    def test_openai_proxy_app(self):
        """Test do_tracking with various dataframe sizes using parametrized test."""
        assert type(self.TestOpenAIProxyApp.openai_proxy_app) == mlrun.runtimes.nuclio.application.application.ApplicationRuntime


