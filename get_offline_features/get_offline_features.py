from typing import Union, List, Dict

import mlrun.feature_store as fs
from mlrun.datastore.targets import get_target_driver, kind_to_driver
from mlrun.datastore.base import DataItem
from mlrun.execution import MLClientCtx


def get_offline_features(
    context: MLClientCtx,
    feature_vector: str,
    entity_rows: DataItem = None,
    entity_timestamp_column: str = None,
    target: Union[str, Dict] = None,
    run_config: Union[str, Dict] = None,
    drop_columns: List[str] = None,
    start_time: str = None,
    end_time: str = None,
    with_indexes: bool = False,
    update_stats: bool = False,
):
    """retrieve offline feature vector results

    specify a feature vector object/uri and retrieve the desired features, their metadata
    and statistics. returns :py:class:`~mlrun.feature_store.OfflineVectorResponse`,
    results can be returned as a dataframe or written to a target

    The start_time and end_time attributes allow filtering the data to a given time range, they accept
    string values or pandas `Timestamp` objects, string values can also be relative, for example:
    "now", "now - 1d2h", "now+5m", where a valid pandas Timedelta string follows the verb "now",
    for time alignment you can use the verb "floor" e.g. "now -1d floor 1H" will align the time to the last hour
    (the floor string is passed to pandas.Timestamp.floor(), can use D, H, T, S for day, hour, min, sec alignment).


    :param context:        MLRun context.
    :param feature_vector: feature vector uri or FeatureVector object.
    :param entity_rows:    URI of the data entity rows to join with.
    :param target:         where to write the results to.
    :param drop_columns:   list of columns to drop from the final result
    :param entity_timestamp_column: timestamp column name in the entity rows dataframe
    :param run_config:     function and/or run configuration
                           see :py:class:`~mlrun.feature_store.RunConfig`
    :param start_time:      datetime, low limit of time needed to be filtered. Optional.
        entity_timestamp_column must be passed when using time filtering.
    :param end_time:        datetime, high limit of time needed to be filtered. Optional.
        entity_timestamp_column must be passed when using time filtering.
    :param with_indexes:    return vector with index columns (default False)
    :param update_stats:    update features statistics from the requested feature sets on the vector. Default is False.

    :returns feature_vector input
    """

    # Preparing entity_rows:
    if entity_rows is not None:
        context.logger.info(f"Creating DataFrame from entity_rows = {entity_rows}")
        entity_rows = entity_rows.as_df()

    # Preparing target:
    if target:
        if isinstance(target, str):
            target = kind_to_driver[target]()

        name = target.name if hasattr(target, "name") else target["name"]
        context.logger.info(f"Preparing '{name}' target")
        target = get_target_driver(target)
    if target.path:
        context.log_result("target", target.path)

    # Preparing run_config:
    if run_config and isinstance(run_config, dict):
        context.logger.info("Preparing run configuration")
        run_config = fs.RunConfig(**run_config)

    # Calling get_offline_features:
    context.logger.info(
        f"getting offline features from the FeatureVector {feature_vector}"
    )
    fs.get_offline_features(
        feature_vector=feature_vector,
        entity_rows=entity_rows,
        entity_timestamp_column=entity_timestamp_column,
        target=target,
        run_config=run_config,
        drop_columns=drop_columns,
        start_time=start_time,
        end_time=end_time,
        with_indexes=with_indexes,
        update_stats=update_stats,
    )

    context.log_result("feature_vector", feature_vector)
