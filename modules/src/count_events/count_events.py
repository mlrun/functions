from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase, ModelMonitoringApplicationMetric,
)
import mlrun.model_monitoring.applications.context as mm_context


class CountApp(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> ModelMonitoringApplicationMetric:
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