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
from dask.distributed import Client

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import ChartArtifact, TableArtifact, PlotArtifact

from typing import IO, AnyStr, Union, List, Optional

def table_summary(
    context: MLClientCtx,
    dask_client: Union[DataItem, str],
    dask_key: str = 'my_dask_dataframe',
    target_path: str = '',
    name: str = 'table_summary.csv',
    key: str = 'table_summary'
) -> None:
    """Summarize a table
    
    :param context:         the function context
    :param dask_client:     path to the dask client scheduler json file, as
                            string or artifact
    :param dask_key:        key of dataframe in dask client 'datasets' attribute
    :param target_path:     destimation folder for table summary file
    :param name:            name of table summary file (with extension like .csv)
    :param key:             key of table summary in artifact store
    """
    print(context.__dict__)
    dask_client = Client(scheduler_file=str(dask_client))
    df = dask_client.get_dataset('dask_key')
    print(df.head())
    dscr = df.describe() 
    
    filepath = os.path.join(target_path, name)
    dd.to_csv(dscr, filepath, single_file=True, index=False)
    context.log_artifact(key, target_path=filepath)
    