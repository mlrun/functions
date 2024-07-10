# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import shutil
from pathlib import Path

import mlrun

METRICS_PATH = "data/metrics.pq"
ARTIFACTS_PATH = "artifacts"
RUNS_PATH = "runs"
SCHEDULES_PATH = "schedules"
PLOTS_PATH = os.path.abspath("./artifacts/feature-selection-feature-selection/0")


def _validate_paths(paths):
    """
    Check if all the expected plot are saved
    """
    base_folder = PLOTS_PATH
    for path in paths:
        full_path = os.path.join(base_folder, path)
        if Path(full_path).is_file():
            print(f"{path} exist")
        else:
            raise FileNotFoundError(f"{path} not found!")
    return True


def _delete_outputs(paths):
    for path in paths:
        if Path(path).is_dir():
            shutil.rmtree(path)


def test_run_local_feature_selection():
    fn = mlrun.import_function("function.yaml")
    run = fn.run(
        params={
            "k": 2,
            "min_votes": 0.3,
            "label_column": "is_error",
        },
        inputs={"df_artifact": "data/metrics.pq"},
        artifact_path="artifacts/",
        local=True,
    )
    assert _validate_paths(
        [
            "chi2.html",
            "f_classif.html",
            "f_regression.html",
            "mutual_info_classif.html",
        ]
    )
    _delete_outputs({ARTIFACTS_PATH, RUNS_PATH, SCHEDULES_PATH})
