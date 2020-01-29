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
    persist: bool = True,
    dask_key: str = 'my_dask_dataframe'
) -> None:
    """Load parquet dataset into dask cluster
    
    If no cluster is found loads a new one and persist the data to it
    """
    # Setup Dask
    if hasattr(context, 'dask_client'):
        dask_client = context.dask_client  
    else:
        context.dask_client = Client(LocalCluster(n_workers=shards,
                                                  threads_per_worker=threads_per))
        context.logger.info(context.dask_client)

    assert context.dask_client
    
    df = dd.read_parquet(parquet_url)

    if persist and context:
        df = context.dask_client.persist(df)
        context.dask_client.datasets[dask_key] = df
        print(df.head())
        # or can use:
        # context.dask_client.publish_dataset(my_dataset=df)
