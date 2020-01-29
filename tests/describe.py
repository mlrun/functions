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

def table_summary(
    context: MLClientCtx,
    dask_key: str = 'my_dask_dataframe',
    target_path: str = '',
    name: str = 'table_summary.csv',
    key: str = 'table_summary'
) -> None:
    """Summarize a table
    """
    if hasattr(context, 'dask_client'):
        dscr = context.dask_client.datasets[dask_key].describe() 
        filepath = os.path.join(target_path, name)
        dscr.to_csv(filepath)
        context.log_artifact(key, target_path=filepath)
    else:
        context.logger.info('no dask_client found')
    