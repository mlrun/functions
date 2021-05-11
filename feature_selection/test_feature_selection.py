from feature_selection import feature_selection
from mlrun import new_task, run_local
from pathlib import Path
import os
import shutil

ARTIFACTS_PATH ='artifacts'


def _validate_paths(paths: {}):
    base_folder = ARTIFACTS_PATH
    for path in paths:
        full_path = os.path.join(base_folder, path)
        if Path(full_path).is_file():
            print("File exist")
        else:
            raise FileNotFoundError


def test_run_local():
    if Path(ARTIFACTS_PATH).is_dir():
        shutil.rmtree(ARTIFACTS_PATH)

    task = new_task(name="task-feature-selection",
                    handler = feature_selection,
                    params={'k': 2,
                           'min_votes': 0.3,
                           'label_column': 'is_error'},
                   inputs={'df_artifact': 'data/metrics.pq'},
                   )
    run_local(task=task,
              artifact_path=os.path.join(os.path.abspath('./'), 'artifacts'))
    _validate_paths({'feature_scores.parquet',
                     'selected_features.parquet'})
