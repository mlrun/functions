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
from mlrun import code_to_function, import_function

DATA_URL = "https://s3.wasabisys.com/iguazio/data/market-palce/arc_to_parquet/higgs-sample.csv.gz"


def test_run_arc_to_parquet():
    fn = code_to_function(
        name="test_arc_to_parquet",
        filename="arc_to_parquet.py",
        handler="arc_to_parquet",
        kind="local",
    )
    run = fn.run(
        params={"key": "higgs-sample"},
        handler="arc_to_parquet",
        inputs={"archive_url": DATA_URL},
        artifact_path="artifacts",
        local=False,
    )

    assert run.outputs["higgs-sample"]


def test_run_local_arc_to_parquet():
    import os

    os.getcwd()
    fn = import_function("function.yaml")
    run = fn.run(
        params={"key": "higgs-sample"},
        handler="arc_to_parquet",
        inputs={"archive_url": DATA_URL},
        artifact_path=os.getcwd() + "/artifacts",
        local=True,
    )

    assert run.outputs["higgs-sample"]
