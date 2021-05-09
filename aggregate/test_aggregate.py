from pathlib import Path
import shutil
from mlrun import code_to_function

METRICS_PATH = 'data/metrics.pq'
ARTIFACTS_PATH = 'artifacts'
RUNS_PATH = 'runs'
SCHEDULES_PATH = 'schedules'
AGGREGATE_PATH = 'artifacts/aggregate.pq'


def _delete_outputs(paths):
    for path in paths:
        if Path(path).is_dir():
            shutil.rmtree(path)


def test_run_local_aggregate():
    fn = code_to_function(name='test_aggregate',
                          filename="aggregate.py",
                          handler="aggregate",
                          kind="job",
                          )
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
    assert Path(AGGREGATE_PATH).is_file()
    _delete_outputs({ARTIFACTS_PATH, RUNS_PATH, SCHEDULES_PATH})
