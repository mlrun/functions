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

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

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
    persist: bool = True
) -> None:
    """Load parquet file or dataset into dask cluster
    
    
    """
    # Setup Dask
    if hasattr(context, 'dask_client'):
        dask_client = context.dask_client  
    else:
        dask_client = Client(LocalCluster(n_workers=shards))
    
    df = dd.read_parquet(parquet_url)

    if persist:
        df = df.persist()
