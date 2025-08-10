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
from mlrun import code_to_function, new_function, import_function
from pathlib import Path
import os

DATA_URL = 'https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv'
ARTIFACTS_PATH = 'artifacts'
PLOTS_PATH = ARTIFACTS_PATH + '/plots'
GENERATED_ARTIFACTS = [
    'correlation', 'correlation-matrix', 'histograms', 'imbalance', 'imbalance-weights-vec', 'violin'
]


def _create_dask_func(uri):
    dask_cluster_name = "dask-cluster"
    dask_cluster = new_function(dask_cluster_name, kind='dask', image='mlrun/ml-models')
    dask_cluster.spec.remote = False
    dask_uri = uri
    dask_cluster.export(dask_uri)


def test_code_to_function_describe_dask():
    print("Starting test_code_to_function_describe_dask")
    dask_uri = "dask_func.yaml"
    _create_dask_func(dask_uri)
    print("Created Dask function")
    fn = code_to_function(filename="describe_dask.py", kind='local')
    fn.spec.command = "describe_dask.py"
    print("About to run function")
    try:
        run = fn.run(
            inputs={"dataset": DATA_URL},
            params={
                'update_dataset': True,
                'label_column': 'label',
                'dask_function': dask_uri,
            },
            handler="summarize",
        )
        print("Function run completed")
        assert all(run.artifact(artifact).get() for artifact in GENERATED_ARTIFACTS)
        print("All artifacts retrieved successfully")
    except Exception as e:
        print(f"Exception in test_code_to_function_describe_dask: {e}")
        raise


def test_import_function_describe_dask():
    print("Starting test_import_function_describe_dask")
    dask_uri = "dask_func.yaml"
    _create_dask_func(dask_uri)
    print("Created Dask function")
    fn = import_function('function.yaml')
    print("Imported function")
    try:
        run = fn.run(
            inputs={
                "dataset": DATA_URL},
            params={
                'update_dataset': True,
                'label_column': 'label',
                'dask_function': dask_uri,
            },
            handler="summarize",
            artifact_path=os.getcwd() + '/artifacts',
            local=True,
        )
        print("Function run completed")
        assert all(run.artifact(artifact).get() for artifact in GENERATED_ARTIFACTS)
        print("All artifacts retrieved successfully")
    except Exception as e:
        print(f"Exception in test_import_function_describe_dask: {e}")
        raise
