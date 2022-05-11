"""Snowflake Dask - Ingest Snaowflake data with Dask"""
import warnings
import mlrun
from mlrun.execution import MLClientCtx
import snowflake.connector as snow
from dask.distributed import Client
from dask.dataframe import from_delayed
from dask import delayed
from dask import dataframe as dd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

warnings.filterwarnings("ignore")

@delayed
def load(batch):

    """A delayed load one batch."""

    try:
        print("BATCHING")
        df_ = batch.to_pandas()
        return df_
    except Exception as e:
        print(f"Failed on {batch} for {e}")
        raise

def load_results(context: MLClientCtx,
                 dask_client: str,
                 connection_info: str,
                 query: str,
                 parquet_out_dir = None,
                 publish_name = None
                ) -> None:

    """Snowflake Dask - Ingest Snaowflake data with Dask

    :param context:           the function context
    :param dask_client:       dask cluster function name
    :param connection_info:   Snowflake database connection info (this will be in a secret later)
    :param query:             query to for Snowflake
    :param parquet_out_dir:   directory path for the output parquet files
                              (default None, not write out)
    :param publish_name:      name of the dask dataframe to publish to the dask cluster
                              (default None, not publish)

    """
    context = mlrun.get_or_create_ctx('snawflake-dask-cluster')
    sf_password = context.get_secret('sfPassword')
    pk_path =  context.get_secret('pkPath')
    pk_password =  context.get_secret('pkPassword')

    if pk_path and pk_password:
        with open(pk_path, "rb") as key:
            p_key= serialization.load_pem_private_key(
                key.read(),
                password=str(pk_password).encode(),
                backend=default_backend()
            )
        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8
            ,encryption_algorithm=serialization.NoEncryption()
        )
        connection_info.pop('password', 'No password found')
        connection_info['private_key'] = pkb
    elif sf_password:
        connection_info['password'] = sf_password
    else:
        raise Exception("\nPlease set up the secret for Snowflake in your project!\n")

    # setup dask client from the MLRun dask cluster function
    if dask_client:
        client = mlrun.import_function(dask_client).client
        context.logger.info(f'Existing dask client === >>> {client}\n')
    else:
        client = Client()
        context.logger.info(f'\nNewly created dask client === >>> {client}\n')

    conn = snow.connect(**connection_info)
    cur = conn.cursor()
    cur.execute(query)
    batches = cur.get_result_batches()
    context.logger.info(f'batches len === {len(batches)}\n')

    dfs = []
    for batch in batches:
        if batch.rowcount > 0:
            df = load(batch)
            dfs.append(df)
    ddf = from_delayed(dfs)

    # materialize the query results set for some sample compute

    ddf_describe = ddf.describe().compute()

    context.logger.info(f'query  === >>> {query}\n')
    context.logger.info(f'ddf  === >>> {ddf}\n')
    context.log_result('number of rows', len(ddf.index))
    context.log_dataset("ddf_describe", df=ddf_describe)

    if publish_name:
        context.log_result('data_set_name', publish_name)
        if not client.list_datasets():
            ddf.persist(name = publish_name)
            client.publish_dataset(publish_name=ddf)

    if parquet_out_dir:
        dd.to_parquet(df=ddf, path=parquet_out_dir)
        context.log_result('parquet directory', parquet_out_dir)
