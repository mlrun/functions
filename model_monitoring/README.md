# Model Monitoring

## Initial set up (and pre-requisites)
1. Make sure you have the `mlrun-api` datasource available in your Grafana instance, otherwise add it by:
   1. Log into your grafana instance
   2. Navigate `Configuration -> Data Sources`
   3. Press `Add data source`
   4. Configure:
   ```
   Name: mlrun-api
   URL: http://mlrun-api:8080/api/grafana-proxy/model-endpoints
   Access: Server (default)
   
   ## Add a custom header of:
   X-V3io-Session-Key: <YOUR ACCESS KEY>
   ```
   5. Press `Save & Test` to make sure it works, a confirmation message should appear when this button is pressed
2. Import the available dashboards `(resource/*.json)` to you Grafana instance
3. To allow the system to utilize drift measurement, make sure you supply the train set when logging the model on the 
   training step
   ```
   # Log model
    context.log_model(
        "model",
        body=dumps(model),
        artifact_path=context.artifact_subpath("models"),
        extra_data=eval_metrics, 
        model_file="model.pkl",
        metrics=context.results,
        training_set=X_test,  # <- make sure this is passed into log_model
        labels={"class": "sklearn.linear_model.LogisticRegression"}
    )
   ```
4. When serving a model, make sure that the Nuclio function is deployed with tracking enabled by applying 
   `fn.set_tracking()`

## Configuration
The stream processing portion of the model monitoring, can be deployed under multiple configuration options. The 
available configurations can be found under `stream.Config`. Once configured it should be supplied as environment 
parameters to the Nuclio function by setting `fn.set_envs`  
```python
    project: str                        # project name
    sample_window: int                  # The sampling window for the data that flows into the TSDB and the KV
    kv_path_template: str               # Path template for the kv table
    tsdb_path_template: str             # Path template for the tsdb table
    parquet_path_template: str          # v3io parquets path template, assumes v3io is mounted
    tsdb_batching_max_events: int       # The max amount of event to batch before writing the batch to tsdb 
    tsdb_batching_timeout_secs: int     # The max amount of seconds a given batch can be gathered before being emitted
    parquet_batching_max_events: int    # The max amount of event to batch before writing the batch to parquet
    parquet_batching_timeout_secs: int  # The max amount of seconds, a given batch can be gathered before being written to parquet
    container: str                      # container name
    v3io_access_key: str                # V3IO Access key
    v3io_framesd: str                   # V3IO framesd URL
    time_format: str                    # The time format into which time related fields will be converted
    aggregate_count_windows: List[str]  # List of window sizes for predictions count
    aggregate_count_period: str         # Period of predictions count windows
    aggregate_avg_windows: List[str]    # List of window sizes for average latency
    aggregate_avg_period: str           # Period of average latency windows
```

