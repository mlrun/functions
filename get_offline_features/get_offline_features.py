from typing import Union, List, Optional, Dict

import pandas as pd
import mlrun.feature_store as fs
from mlrun.datastore.targets import get_target_driver
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
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        with_indexes: bool = False,
        update_stats: bool = False,
):
    """retrieve offline feature vector results

    specify a feature vector uri and retrieve the desired features, their metadata
    and statistics. returns :py:class:`~mlrun.feature_store.OfflineVectorResponse`,
    results can be returned as a dataframe or written to a target

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
    # --- Preparing inputs ---

    # Preparing entity_rows:
    if entity_rows is not None:
        context.logger.info(f'Preparing entity_rows: {entity_rows}')
        entity_rows = entity_rows.as_df()

    # Preparing target:
    if target:
        context.logger.info(f'Preparing target: {target}')
        target = get_target_driver(target)
    # Preparing run_config:
    if isinstance(run_config, dict):
        context.logger.info('Preparing run configuration')
        run_config = fs.RunConfig(**run_config)

    context.logger.info(f'getting offline features from the FeatureVector {feature_vector}')
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
        update_stats=update_stats
    )

    return feature_vector
