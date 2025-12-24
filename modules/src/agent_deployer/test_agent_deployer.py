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

import unittest
from unittest.mock import patch, MagicMock
from agent_deployer import AgentDeployer
import mlrun.errors



class TestAgentDeployer(unittest.TestCase):

    def setUp(self):
        # Common parameters for a minimal AgentDeployer instance
        self.deployer_params = {
            "agent_name": "test_agent",
            "model_class_name": "MyModelClass",
            "function": "path/to/my_function_file.py",
        }
        self.deployer = AgentDeployer(**self.deployer_params)

    # --- Test Cases for Properties ---

    @patch('agent_deployer.get_current_project')  # Patch the import in the *module* you are testing
    def test_project_property_returns_project(self, mock_get_current_project):
        """Test that the project property returns the project if it exists."""
        mock_proj = MagicMock()
        mock_get_current_project.return_value = mock_proj

        self.assertEqual(self.deployer.project, mock_proj)
        mock_get_current_project.assert_called_once_with(silent=True)

    @patch('agent_deployer.get_current_project', return_value=None)
    def test_project_name_raises_error_if_no_project(self, mock_get_current_project):
        """Test that project_name raises an error when no project is found."""
        with self.assertRaises(mlrun.errors.MLRunInvalidArgumentError):
            _ = self.deployer.project_name

    @patch('agent_deployer.get_current_project')
    def test_project_name_returns_name(self, mock_get_current_project):
        """Test that project_name correctly retrieves the name from the project metadata."""
        mock_proj = MagicMock()
        mock_proj.metadata.name = "test-project-name"
        mock_get_current_project.return_value = mock_proj

        self.assertEqual(self.deployer.project_name, "test-project-name")


    @patch('agent_deployer.AgentDeployer.project', new_callable=unittest.mock.PropertyMock)
    def test_configure_model_monitoring_handles_conflict_error(self, mock_project_prop):
        """Test that the method handles expected exceptions during enable_model_monitoring."""
        mock_project = MagicMock()
        # Simulate an expected error that should be caught and passed over
        mock_project.enable_model_monitoring.side_effect = mlrun.errors.MLRunConflictError("Already deployed")
        mock_project_prop.return_value = mock_project

        # This should run without raising an uncaught exception
        self.deployer.configure_model_monitoring()
        mock_project.enable_model_monitoring.assert_called_once()