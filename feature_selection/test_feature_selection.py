from mlrun import code_to_function
from pathlib import Path
import os
import shutil

METRICS_PATH = 'data/metrics.pq'
ARTIFACTS_PATH = 'artifacts'
RUNS_PATH = 'runs'
SCHEDULES_PATH = 'schedules'


def _delete_outputs(paths):
    for path in paths:
        if Path(path).is_dir():
            shutil.rmtree(path)


def _validate_paths(paths: {}):
    base_folder = ARTIFACTS_PATH
    for path in paths:
        full_path = os.path.join(base_folder, path)
        if Path(full_path).is_file():
            print("File exist")
        else:
            raise FileNotFoundError


def test_run_local_feature_selection():
    fn = code_to_function(name='test_run_local_feature_selection',
                          filename="feature_selection.py",
                          handler="feature_selection",
                          kind="local",
                          )
    fn.spec.command = "feature_selection.py"
    fn.run(params={'k': 2,
                           'min_votes': 0.3,
                           'label_column': 'is_error'},
           # , local=True
           inputs={'df_artifact': 'data/metrics.pq'},
           artifact_path='artifacts/'
           )
    _validate_paths({'feature_scores.parquet',
                     'selected_features.parquet'})
    _delete_outputs({ARTIFACTS_PATH, RUNS_PATH, SCHEDULES_PATH})