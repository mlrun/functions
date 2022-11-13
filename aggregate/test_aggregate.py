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
from pathlib import Path
from mlrun import code_to_function, import_function


AGGREGATE_PATH = "artifacts/aggregate.pq"
DATA = "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq"


def test_run_local_aggregate():
    fn = code_to_function(name='test_aggregate',
                          filename="aggregate.py",
                          handler="aggregate",
                          kind="local",
                          )
    fn.spec.command = 'aggregate.py'
    run = fn.run(
        params={
            'metrics': ['cpu_utilization'],
            'labels': ['is_error'],
            'metric_aggs': ['mean', 'sum'],
            'label_aggs': ['max'],
            'suffix': 'daily',
            'inplace': False,
            'window': 5,
            'center': True,
            'save_to': AGGREGATE_PATH,
            'files_to_select': 2
        },
        #  local=True,
        inputs={'df_artifact': DATA}
    )
    assert run.artifact('aggregate').get()


def test_import_function_aggregate():
    fn = import_function("function.yaml")
    run = fn.run(
        params={
            'metrics': ['cpu_utilization'],
            'labels': ['is_error'],
            'metric_aggs': ['mean', 'sum'],
            'label_aggs': ['max'],
            'suffix': 'daily',
            'inplace': False,
            'window': 5,
            'center': True,
            'save_to': AGGREGATE_PATH,
            'files_to_select': 2,
        },
        local=True,
        inputs={'df_artifact': DATA},
    )
    assert run.artifact('aggregate').get()
