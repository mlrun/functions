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
from mlrun import import_function
import os


ARTIFACT_PATH = "artifacts"
FUNCTION_PATH = "functions"
PLOTS_PATH = "plots"
RUNS_PATH = "runs"
SCHEDULES_PATH = "schedules"


def test_local_xgb_custom():
    fn = import_function("function.yaml")
    run = fn.run(
        params={
            "nrows": 8192,
            "label_type": "float",
            "local_path": "./artifacts/inputs/xgb_custom",
        },
        handler="gen_outliers",
        local=True,
    )

    run = fn.run(
        params={
            "num_boost_round": 40,
            "verbose_eval": False,
            "XGB_max_depth": 2,
            "XGB_subsample": 0.9,
            "test_set_key": "test-set",
        },
        inputs={"dataset": run.artifact('xgb-outs').url},
        handler="fit",
        local=True,
    )
    assert run.artifact('learning-curves').get()
