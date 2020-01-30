# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from typing import IO, AnyStr, Union, List, Optional

def parquet_to_dask(
    context: MLClientCtx,
    parquet_url: Union[DataItem, str, Path, IO[AnyStr]],
    inc_cols: Optional[List[str]] = None,
    index_cols: Optional[List[str]] = None,
    shards: int = 4,
    threads_per: int = 4,
    processes: bool = False,
    memory_limit: str = '2GB',
    persist: bool = True,
    dask_key: str = 'my_dask_dataframe',
    target_path: str = ''
) -> None:
    """Load parquet dataset into dask cluster
    
    If no cluster is found loads a new one and persist the data to it
    """
    if hasattr(context, 'dask_client'):
        context.logger.info('found cluster...')
        dask_client = context.dask_client
    else:
        context.logger.info('starting new cluster...')
        cluster = LocalCluster(n_workers=shards,
                               threads_per_worker=threads_per,
                               processes=processes,
                               memory_limit=memory_limit)
        dask_client = Client(cluster)
    
    context.logger.info(dask_client)
 
    df = dd.read_parquet(parquet_url)

    if persist and context:
        df = dask_client.persist(df)
        dask_client.publish_dataset(dask_key=df)
        context.dask_client = dask_client
        
        # share the scheduler
        filepath = os.path.join(target_path, 'scheduler.json')
        dask_client.write_scheduler_file(filepath)
        context.log_artifact('scheduler', target_path=filepath)
        
        print(df.head())
