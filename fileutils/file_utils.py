# Copyright 2019 Iguazio
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
import zipfile
import json
from tempfile import mktemp
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from mlrun.execution import MLClientCtx
from typing import IO, AnyStr, Union, List

def open_archive(context: MLClientCtx, 
                 target_path: str = '',
                 key: str = 'content',
                 archive_url: = ''):
    """Open a file/object archive into a target directory.

    The file contents will be available within target_dir, and
    and target_dir will be logged as an artifact under the key
    specified (defaults to 'content').

    :param context:      function context
    :param target_path:  destination for file artifact
    :param key:          key of item in artifact store
    :param archive_url:  source archive url
    """
    # Define locations
    os.makedirs(target_path, exist_ok=True)
    context.logger.info('verified directories')
    
    # Extract dataset from zip
    context.logger.info('extracting zip')
    zip_ref = zipfile.ZipFile(archive_url, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()
    
    context.logger.info(f'extracted archive to {target_dir}')
    context.log_artifact(key, target_path=target_dir)
    
import os


def arc_to_parquet(
    context: MLClientCtx,
    archive_url: Union[str, Path, IO[AnyStr]],
    header: Union[None, List[str]] = None,
    target_path: str = "",
    name: str = "",
    chunksize: int = 10_000,
    log_data: bool = True,
    add_uid: bool = False,
    key: str = 'raw_data'
) -> None:
    """Open a file/object archive and save as a parquet file.
    
    :param context:     function context
    :param archive_url: any valid string path consistent with the path variable
                        of pandas.read_csv. ncluding strings as file paths, as urls, 
                        pathlib.Path objects, etc...
    :param header:      column names
    :param target_path: destination folder of table
    :param name:        name file to be saved locally, also
    :param chunksize:   (0) row size retrieved per iteration
    :param log_data:    (True) if True, log the data so that it is available
                        at the next step
    :param add_uid:     (False) add the metadata uid to the target_path so that 
                        runs can be identified
    :param key:         key in artifact store (when log_data=True)
    """
    if not name.endswith(".parquet"):
        name += ".parquet"
    
    if not add_uid:
        uid = ''
    else:
        uid = context.uid
        
    dest_path = os.path.join(target_path, uid, name)
    os.makedirs(os.path.join(target_path, uid), exist_ok=True)
    
    if not os.path.isfile(dest_path):
        context.logger.info("destination file does not exist, downloading")
        pqwriter = None
        for i, df in enumerate(
            pd.read_csv(archive_url, chunksize=chunksize, names=header)
        ):
            table = pa.Table.from_pandas(df)
            if i == 0:
                pqwriter = pq.ParquetWriter(dest_path, table.schema)
            pqwriter.write_table(table)

        if pqwriter:
            pqwriter.close()

        context.logger.info(f"saved table to {dest_path}")
    else:
        context.logger.info("destination file already exists")

    if log_data:
        context.logger.info(f"assign data to {key} in artifact store")
        context.log_artifact(key, target_path=dest_path)