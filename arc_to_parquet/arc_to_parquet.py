import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from typing import List


def _chunk_readwrite(
        archive_url,
        dest_path,
        chunksize,
        header,
        encoding,
        dtype
):
    """stream read and write archives

    pandas reads and parquet writes

    notes
    -----
    * dest_path can be either a file.parquet, or in hte case of partitioned parquet
      it will be only the destination folder of the parquet partition files
    """
    pqwriter = None
    header = []
    for i, df in enumerate(pd.read_csv(archive_url, chunksize=chunksize,
                                       names=header, encoding=encoding,
                                       dtype=dtype)):
        table = pa.Table.from_pandas(df)
        if i == 0:
            if dataset:
                header = copy(table.schema)
            else:
                pqwriter = pq.ParquetWriter(dest_path, table.schema)
        if dataset:
            pq.write_to_dataset(table, root_path=dest_path, partition_cols=partition_cols)
        else:
            pqwriter.write_table(table)
    if pqwriter:
        pqwriter.close()

    return header


def arc_to_parquet(
        context: MLClientCtx,
        archive_url: DataItem,
        header: List[str] = [None],
        chunksize: int = 0,
        dtype=None,
        encoding: str = "latin-1",
        key: str = "data",
        dataset: str = "None",
        part_cols=[],
        file_ext: str = "parquet",
        index: bool = False,
        refresh_data: bool = False,
        stats: bool = False
) -> None:
    """Open a file/object archive and save as a parquet file or dataset

    Notes
    -----
    * this function is typically for large files, please be sure to check all settings
    * partitioning requires precise specification of column types.
    * the archive_url can be any file readable by pandas read_csv, which includes tar files
    * if the `dataset` parameter is not empty, then a partitioned dataset will be created
    instead of a single file in the folder `dataset`
    * if a key exists already then it will not be re-acquired unless the `refresh_data` param
    is set to `True`.  This is in case the original file is corrupt, or a refresh is
    required.

    :param context:        the function context
    :param archive_url:    MLRun data input (DataItem object)
    :param chunksize:      (0) when > 0, row size (chunk) to retrieve
                           per iteration
    :param dtype           destination data type of specified columns
    :param encoding        ("latin-8") file encoding
    :param key:            key in artifact store (when log_data=True)
    :param dataset:        (None) if not None then "target_path/dataset"
                           is folder for partitioned files
    :param part_cols:      ([]) list of partitioning columns
    :param file_ext:       (parquet) csv/parquet file extension
    :param index:          (False) pandas save index option
    :param refresh_data:   (False) overwrite existing data at that location
    :param stats:          (None) calculate table stats when logging artifact
    """
    base_path = context.artifact_path
    os.makedirs(base_path, exist_ok=True)

    archive_url = archive_url.local()

    if dataset is not None:
        dest_path = os.path.join(base_path, dataset)
        exists = os.path.isdir(dest_path)
    else:
        dest_path = os.path.join(base_path, key + f".{file_ext}")
        exists = os.path.isfile(dest_path)

    if not exists:
        context.logger.info("destination file does not exist, downloading")
        if chunksize > 0:
            header = _chunk_readwrite(archive_url, dest_path, chunksize,
                                      encoding, dtype)
            context.log_dataset(key=key, stats=stats, format='parquet',
                                target_path=dest_path)
        else:
            df = pd.read_csv(archive_url)
            context.log_dataset(key, df=df, format=file_ext, index=index)
    else:
        context.logger.info("destination file already exists, nothing done")