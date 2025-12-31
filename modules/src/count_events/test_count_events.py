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


from datetime import datetime
from unittest.mock import Mock

import mlrun.model_monitoring.applications.context as mm_context
import pandas as pd
import pytest
from count_events import CountApp
from mlrun.model_monitoring.applications import ModelMonitoringApplicationMetric


class TestCountApp:
    """Test suite for CountApp class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.count_app = CountApp()

    @staticmethod
    def _create_mock_monitoring_context(sample_df, model_endpoint_name="test-model"):
        """Helper method to create a mock monitoring context."""
        mock_context = Mock(spec=mm_context.MonitoringApplicationContext)

        # Mock the sample dataframe
        mock_context.sample_df = sample_df

        # Mock the logger
        mock_logger = Mock()
        mock_context.logger = mock_logger

        # Mock the model endpoint
        mock_model_endpoint = Mock()
        mock_model_endpoint.metadata.name = model_endpoint_name
        mock_context.model_endpoint = mock_model_endpoint

        # Mock time attributes
        mock_context.start_infer_time = datetime(2025, 1, 1, 10, 0, 0)
        mock_context.end_infer_time = datetime(2025, 1, 1, 11, 0, 0)

        return mock_context

    @pytest.mark.parametrize("df_size", [0, 1, 10, 100, 1000])
    def test_do_tracking_with_various_dataframe_sizes(self, df_size):
        """Test do_tracking with various dataframe sizes using parametrized test."""
        # Arrange
        if df_size == 0:
            test_df = pd.DataFrame()
        else:
            test_df = pd.DataFrame({"col1": range(df_size)})

        mock_context = self._create_mock_monitoring_context(test_df)

        # Act
        result = self.count_app.do_tracking(mock_context)

        # Assert
        assert isinstance(result, ModelMonitoringApplicationMetric)
        assert result.value == df_size
        assert result.name == "count"
