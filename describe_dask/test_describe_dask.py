from describe_dask import summarize
from mlrun import code_to_function, new_function
from pathlib import Path
import shutil
import mlrun
from os import path
from dask.distributed import Client
import os

DATA_URL = 'https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv'
PLOTS_PATH ='plots'


def _validate_paths(paths: {}):
    base_folder = PLOTS_PATH
    for path in paths:
        full_path = os.path.join(base_folder, path)
        if Path(full_path).is_file():
            print("File exist")
        else:
            raise FileNotFoundError


# def test_describe_dask_local():
#     project_name = "dask-demo"
#     project_path = path.abspath(project_name)
#     mlrun.set_environment(project=project_name, artifact_path='./', api_path=project_path)
#     dask_cluster_name = "dask-cluster"
#     dask_cluster = new_function(dask_cluster_name, kind='dask', image='mlrun/ml-models')
#     # function URI is db://<project>/<name>
#     #dask_uri = f'db://{project_name}/{dask_cluster_name}'
#     dask_uri = "dask-demo/functions/dask-demo/test-describe-dask/latest.yaml"
#     #client = mlrun.import_function(dask_ri).client
#     if Path(PLOTS_PATH).is_dir():
#         shutil.rmtree(PLOTS_PATH)
#     fn = code_to_function(name='test_describe_dask',
#                           filename="describe_dask.py",
#                           handler="summarize",
#                           kind="local",
#                           )
#     fn.spec.command = "describe_dask.py"
#     fn.run(inputs={"table": DATA_URL},
#                     params={'update_dataset': True,
#                             'label_column': 'label'
#                             ,'dask_function': dask_uri
#
#                             }
#                    ,handler = "summarize"
#
#             ,artifact_path='artifacts'
#            )
#     # task = new_task(name="task-describe",
#     #                 handler=summarize,
#     #                 )
#     # run_local(task)
#     _validate_paths({'corr.html',
#                      'correlation-matrix.csv',
#                      'hist.html',
#                      'imbalance.html',
#                      'imbalance-weights-vec.csv',
#                      'violin.html'})
