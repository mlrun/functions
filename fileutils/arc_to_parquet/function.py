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
import pyarrow.parquet as pq
import pyarrow as pa
from pickle import dump, load

from mlrun.execution import MLClientCtx
from typing import IO, AnyStr, Union, List, Optional


def arc_to_parquet(
    context: MLClientCtx,
    archive_url: Union[str, Path, IO[AnyStr]],
    header: Optional[List[str]] = None,
    inc_cols: Optional[List[str]] = None,
    target_path: str = "",
    name: str = "",
    chunksize: int = 10_000,
    dtype=None,
    encoding: str = 'latin-1',
    key: str = 'data',
    dataset: Optional[str] = None,
    partition_cols = [],
) -> None:
    """Open a file/object archive and save as a parquet file.
    
    Partitioning requires precise specification of column types.
    
    :param context:     function context
    :param archive_url: any valid string path consistent with the path variable
                        of pandas.read_csv, including strings as file paths, as urls, 
                        pathlib.Path objects, etc...
    :param header:      column names
    :param inc_cols:    include only these columns
    :param target_path: destination folder of table
    :param name:        name file to be saved locally, also
    :param chunksize:   (0) row size retrieved per iteration
    :param dtype        destination data type of specified columns
    :param encoding     ('latin-8') file encoding
    :param key:         key in artifact store (when log_data=True)
    :param dataset:     (None) if not None then 'target_path/dataset'
                        is folder for partitioned files
    :param part_cols:   ([]) list of partitioning columns
    
    """
    if not name.endswith(".pqt"):
        name += ".pqt"
    
    if dataset is not None:
        os.makedirs(os.path.join(target_path, dataset), exist_ok=True)
        dest_path = os.path.join(target_path, dataset)
    else:
        os.makedirs(os.path.join(target_path), exist_ok=True)
        dest_path = os.path.join(target_path, name)
        
    if not os.path.isfile(dest_path):
        context.logger.info("destination file does not exist, downloading")
        pqwriter = None
        for i, df in enumerate(pd.read_csv(archive_url, 
                                           chunksize=chunksize, 
                                           names=header,
                                           encoding=encoding, 
                                           usecols=inc_cols, 
                                           dtype=dtype)):
            table = pa.Table.from_pandas(df)
            if i == 0:
                filepath = os.path.join(target_path,'header-only.pqt')
                if dataset:
                    # just write header here
                    pq.ParquetWriter(filepath, table.schema)
                    #context.log_artifact('header', target_path=filepath)
                else:
                    # start writing file
                    #context.log_artifact('header', target_path=filepath)
                    pqwriter = pq.ParquetWriter(dest_path, table.schema)
                
            if dataset:
                pq.write_to_dataset(table, root_path=dest_path, partition_cols=partition_cols)
            else:
                pqwriter.write_table(table)
            
        if pqwriter:
            pqwriter.close()

        context.logger.info(f"saved table to {dest_path}")
    else:
        context.logger.info("destination file already exists")
    
    context.log_artifact(key, target_path=dest_path)
