from pathlib import Path
import shutil
from mlrun import code_to_function
import mlrun
import re
import subprocess
from pygit2 import Repository


METRICS_PATH = 'data/metrics.pq'
ARTIFACTS_PATH = 'artifacts'
RUNS_PATH = 'runs'
SCHEDULES_PATH = 'schedules'
AGGREGATE_PATH = 'artifacts/aggregate.pq'


def _delete_outputs(paths):
    for path in paths:
        if Path(path).is_dir():
            shutil.rmtree(path)


def _set_mlrun_hub_url(repo_name = None, branch_name = None, function_name = None):
    repo_name =  re.search("\.com/.*?/", str(subprocess.run(['git', 'remote', '-v'], stdout=subprocess.PIPE).stdout)).group()[5:-1] if not repo_name else repo_name
    branch_name = Repository('.').head.shorthand if not branch_name else branch_name
    function_name = "" if not function_name else function_name # MUST ENTER FUNCTION NAME !!!!
    hub_url = f"https://raw.githubusercontent.com/{repo_name}/functions/{branch_name}/{function_name}/function.yaml"
    mlrun.mlconf.hub_url = hub_url

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


def test_run_imported_aggregate():
    _set_mlrun_hub_url(function_name="aggregate")
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
    assert Path(AGGREGATE_PATH).is_file()
    _delete_outputs({ARTIFACTS_PATH, RUNS_PATH, SCHEDULES_PATH})

