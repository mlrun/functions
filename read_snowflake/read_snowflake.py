import os
from mlrun.execution import MLClientCtx
from pyspark.sql import SparkSession

def read_snowflake(context: MLClientCtx, 
                   url: str, 
                   user: str, 
                   password:str, 
                   db:str, 
                   schema:str, 
                   warehouse:str, 
                   query:str, 
                   target:str) -> None:
    """
    Read Snowflake and write results to parquet
    :param context: MLRun function context
    :param url: Snowflake URL secret
    :param user: Snowflake user secret
    :param password: Snowflake user password secret
    :param db: Snowflake database
    :param schema: Snowflake schema
    :param warehouse: Snowflake warehouse instance eg: "compute_wh"
    :param query: Snowflake query string eg: "select * from customer"
    :param target: Full path to parquet output eg: "v3io://bigdata/output.parquet"
    """
    spark = SparkSession.builder.appName("read_snowflake").getOrCreate()

    sfURL = context.get_secret(url)
    sfUser = context.get_secret(user)
    sfPassword = context.get_secret(password)
    
    sfOptions = {
      "sfURL" : sfURL,
      "sfUser" : sfUser,
      "sfPassword" : sfPassword,
      "sfDatabase" : db,
      "sfSchema" : schema,
      "sfWarehouse" : warehouse,
      "application" : f"Iguazio-{os.getenv('SNOWFLAKE_APPLICATION', 'application')}"
    }

    df = spark.read.format("net.snowflake.spark.snowflake") \
      .options(**sfOptions) \
      .option("query",  query) \
      .load()
    
    df.write.parquet(target)