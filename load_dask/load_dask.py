from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from typing import List, Optional


def load_dask(
        context: MLClientCtx,
        src_data: DataItem,
        dask_key: str = "dask_key",
        inc_cols: Optional[List[str]] = None,
        index_cols: Optional[List[str]] = None,
        dask_persist: bool = True,
        refresh_data: bool = True,
        scheduler_key: str = "scheduler"
) -> None:
    """Load dataset into an existing dask cluster

    dask jobs define the dask client parameters at the job level, this method will raise an error if no client is detected.

    :param context:         the function context
    :param src_data:        url of the data file or partitioned dataset as either
                            artifact DataItem, string, or path object (similar to
                            pandas read_csv)
    :param dask_key:        destination key of data on dask cluster and artifact store
    :param inc_cols:        include only these columns (very fast)
    :param index_cols:      list of index column names (can be a long-running process)
    :param dask_persist:    (True) should the data be persisted (through the `client.persist` op)
    :param refresh_data:    (False) if the dask_key already exists in the dask cluster, this will
                            raise an Exception.  Set to True to replace the existing cluster data.
    :param scheduler_key:   (scheduler) the dask scheduler configuration, json also logged as an artifact
    """
    if hasattr(context, "dask_client"):
        dask_client = context.dask_client
    else:
        raise Exception("a dask client was not found in the execution context")

    df = src_data.as_df(df_module=dd)

    if dask_persist:
        df = dask_client.persist(df)
        if dask_client.datasets and dask_key in dask_client.datasets:
            dask_client.unpublish_dataset(dask_key)
        dask_client.publish_dataset(df, name=dask_key)

    if context:
        context.dask_client = dask_client

    # share the scheduler, whether data is persisted or not
    dask_client.write_scheduler_file(scheduler_key + ".json")

    # we don't use log_dataset here until it can take into account
    # dask origin and apply dask describe.
    context.log_artifact(scheduler_key, local_path=scheduler_key + ".json")