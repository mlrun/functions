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

import mlrun.model_monitoring.applications.context as mm_context
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationMetric,
)


class CountApp(ModelMonitoringApplicationBase):
    """
    Model Monitoring Application that counts the number of events in the given time window.
    """

    def do_tracking(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> ModelMonitoringApplicationMetric:
        """ "
        he do_tracking method implementation for the CountApp class.
        It counts the number of events in the sample data-frame and logs the count.

        :param monitoring_context: The monitoring application context. It includes the current window data as a
                                   pandas data-frame: monitoring_context.sample_df.
        """
        sample_df = monitoring_context.sample_df
        monitoring_context.logger.debug("Sample data-frame", sample_df=sample_df)
        count = len(sample_df)
        monitoring_context.logger.info(
            "Counted events for model endpoint window",
            model_endpoint_name=monitoring_context.model_endpoint.metadata.name,
            count=count,
            start=monitoring_context.start_infer_time,
            end=monitoring_context.end_infer_time,
        )
        return ModelMonitoringApplicationMetric(
            name="count",
            value=count,
        )
