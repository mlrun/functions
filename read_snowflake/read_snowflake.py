from pyspark.sql import SparkSession

def read_snowflake(config, db, schema, warehouse, query, target):
    """
    Read Snowflake and write results to parquet
    :param db: Snowflake database
    :param schema: Snowflake schema
    :param warehouse: Snowflake warehouse instance
    :param query: Snowflake query string
    :param target: Full path to parquet output eg: "v3io://bigdata/output.parquet"
    """
    spark = SparkSession.builder.appName("read_snowflake").getOrCreate()

    sfURL = config["sfURL"]
    sfUser = config["sfUser"]
    sfPassword = config["sfPassword"]
    
    sfOptions = {
      "sfURL" : sfURL,
      "sfUser" : sfUser,
      "sfPassword" : sfPassword,
      "sfDatabase" : db,
      "sfSchema" : schema,
      "sfWarehouse" : warehouse
    }

    df = spark.read.format("net.snowflake.spark.snowflake") \
      .options(**sfOptions) \
      .option("query",  query) \
      .load()
    
    df.write.parquet(target)