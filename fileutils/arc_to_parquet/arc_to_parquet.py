import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from mlrun.execution import MLClientCtx
from typing import IO, AnyStr, Union, List, Optional


def arc_to_parquet(
    context: MLClientCtx,
    archive_url: Union[str, Path, IO[AnyStr]],
    header: Optional[List[str]] = None,
    target_path: str = "",
    name: str = "",
    chunksize: int = 10_000,
    log_data: bool = True,
    add_uid: bool = False,
    key: str = "raw_data",
) -> None:
    """Open a file/object archive and save as a parquet file.
    
    :param context:     function context
    :param archive_url: any valid string path consistent with the path variable
                        of pandas.read_csv, including strings as file paths, as urls, 
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
        uid = ""
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
