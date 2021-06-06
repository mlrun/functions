from describe_dask import summarize
from mlrun import new_task, run_local
from pathlib import Path
import os
import shutil

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
#     if Path(PLOTS_PATH).is_dir():
#         shutil.rmtree(PLOTS_PATH)
#     task = new_task(name="task-describe",
#                     handler=summarize,
#                     inputs={"table": DATA_URL},
#                     params={'update_dataset': True,
#                             'label_column': 'label',
#                             'dask_function': 'db://default/dask_tests'})
#     run_local(task)
#     _validate_paths({'corr.html',
#                      'correlation-matrix.csv',
#                      'hist.html',
#                      'imbalance.html',
#                      'imbalance-weights-vec.csv',
#                      'violin.html'})
