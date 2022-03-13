from pathlib import Path
from mlrun import code_to_function, import_function
import os
import shutil

AGGREGATE_PATH = "artifacts/aggregated_metrics.pq"
DATA = "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq"


def test_run_local_aggregate():
    fn = code_to_function(name='test_aggregate',
                          filename="aggregate.py",
                          handler="aggregate",
                          kind="local",
                          )
    fn.spec.command = 'aggregate.py'
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
           #, local=True
           , inputs={'df_artifact': DATA},
           artifact_path = os.getcwd()
           )
    assert Path(os.getcwd() + '/' + AGGREGATE_PATH).is_file()
    shutil.rmtree(os.getcwd() + '/' + os.path.dirname(AGGREGATE_PATH))

def test_import_function_aggregate():
    fn = import_function("function.yaml")
    fn.run(params={'metrics': ['cpu_utilization'],
                   'labels': ['is_error'],
                   'metric_aggs': ['mean', 'sum'],
                   'label_aggs': ['max'],
                   'suffix': 'daily',
                   'inplace': True,
                   'window': 5,
                   'center': True,
                   'save_to': AGGREGATE_PATH,
                   'files_to_select': 2}
           , local=True
           , inputs={'df_artifact': DATA},
           artifact_path = os.getcwd()
           )
    assert Path(os.getcwd() + '/' + AGGREGATE_PATH).is_file()
    shutil.rmtree(os.getcwd() + '/' + os.path.dirname(AGGREGATE_PATH))