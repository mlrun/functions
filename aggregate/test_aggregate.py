from pathlib import Path
from mlrun import code_to_function
import mlrun
from functions.cli.helpers import set_mlrun_hub_url, delete_outputs
import pandas as pd

METRICS_PATH = 'data/metrics.pq'
ARTIFACTS_PATH = 'artifacts'
RUNS_PATH = 'runs'
SCHEDULES_PATH = 'schedules'
FUNCTION_PATH = 'functions'
AGGREGATE_PATH = 'artifacts/aggregate.pq'


def test_local_aggregate():
    fn = mlrun.import_function("function.yaml")
    fn.run(params={'metrics': ['cpu_utilization'],
                   'labels': ['is_error'],
                   'metric_aggs': ['mean', 'sum'],
                   'label_aggs': ['max'],
                   'suffix': 'daily',
                   'inplace': False,
                   'window': 5,
                   'center': True,
                   'save_to': AGGREGATE_PATH,
                   'files_to_select': 2}
           , local=True
           , inputs={'df_artifact': METRICS_PATH}
           )
    df = pd.read_parquet("data/metrics.pq")
    aggregated_df = pd.read_parquet("artifacts/aggregate.pq")
    assert(aggregated_df["cpu_utilization_mean_daily"].tolist() == df["cpu_utilization"].rolling(5).mean().dropna().tolist()) is True
    assert(aggregated_df["cpu_utilization_sum_daily"].tolist() == df["cpu_utilization"].rolling(5).sum().dropna().tolist()) is True
    # assert(aggregated_df["is_error_max_daily"].tolist() == df["cpu_utilization"].rolling(5).max().dropna().tolist()) is True
    # last test is very slow !!
    delete_outputs({ARTIFACTS_PATH, RUNS_PATH, SCHEDULES_PATH,FUNCTION_PATH})


def test_hub_imported_aggregate():
    set_mlrun_hub_url(function_name="aggregate")
    fn = mlrun.import_function("hub://aggregate")
    fn.run(params={'metrics': ['cpu_utilization'],
                   'labels': ['is_error'],
                   'metric_aggs': ['mean', 'sum'],
                   'label_aggs': ['max'],
                   'suffix': 'daily',
                   'inplace': False,
                   'window': 5,
                   'center': True,
                   'save_to': AGGREGATE_PATH,
                   'files_to_select': 2}
           , local=True
           , inputs={'df_artifact': METRICS_PATH}
           )
    df = pd.read_parquet("data/metrics.pq")
    aggregated_df = pd.read_parquet("artifacts/aggregate.pq")
    assert(aggregated_df["cpu_utilization_mean_daily"].tolist() == df["cpu_utilization"].rolling(5).mean().dropna().tolist()) is True
    assert(aggregated_df["cpu_utilization_sum_daily"].tolist() == df["cpu_utilization"].rolling(5).sum().dropna().tolist()) is True
    # assert(aggregated_df["is_error_max_daily"].tolist() == df["cpu_utilization"].rolling(5).max().dropna().tolist()) is True
    # last test is very slow !!
    delete_outputs({ARTIFACTS_PATH, RUNS_PATH, SCHEDULES_PATH,FUNCTION_PATH})

