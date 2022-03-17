# **Data Preperation Function**

## `Snowflake_dask`

This function query the data from a snowflake database and process the results set 
in parallel in a Dask cluster. 
It will publish the dask dataframe in the cluster for other process to use.
It can also write the result set to parquet files.

```markdown

:param context:           the function context
:param dask_client:       dask cluster function name
:param connection_info:   Snowflake database connection info (this wikk be in a secret later)
:param query:             query to for Snowflake
:param parquet_out_dir:   directory path for the output parquet files (default None, not write out)
:param publish_name:      name of the dask dataframe to publish to the dask cluster (default None, not publish)
```