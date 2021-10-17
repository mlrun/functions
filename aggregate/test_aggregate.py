from pathlib import Path
import shutil
from mlrun import code_to_function, import_function
#Testing the CI with dummy commit 9
AGGREGATE_PATH = "artifacts/aggregate.pq"
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
           , inputs={'df_artifact': DATA}
           )
    assert Path(AGGREGATE_PATH).is_file()


def test_import_function_aggregate():
    fn = import_function("function.yaml")
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
           , inputs={'df_artifact': DATA}
           )
    assert Path(AGGREGATE_PATH).is_file()
