import pandas as pd
import pandas_profiling

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem


def pandas_profiling_report(
    context: MLClientCtx,
    data: DataItem,
) -> None:
    """Create a Pandas Profiling Report for a dataset.
    :param context:         the function context
    :param data:            Dataset to create report for
    """

    # Load dataset
    df = data.as_df()

    # Create Pandas Profiling Report
    profile = df.profile_report(title="Pandas Profiling Report")

    # Save to MLRun DB
    context.log_artifact(
        "Pandas Profiling Report",
        body=profile.to_html(),
        local_path="pandas_profiling_report.html",
    )
