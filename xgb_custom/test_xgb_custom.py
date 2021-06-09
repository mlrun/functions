from mlrun import import_function
import os
from functions.cli.helpers import delete_outputs,set_mlrun_hub_url


ARTIFACT_PATH = "artifacts"
FUNCTION_PATH = "functions"
PLOTS_PATH = "plots"
RUNS_PATH = "runs"
SCHEDULES_PATH = "schedules"


def test_local_xgb_custom():
    fn = import_function("function.yaml")
    fn.run(params={"nrows": 8192,
                   "label_type": "float",
                   "local_path": "./artifacts/inputs/xgb_custom"},
           handler="gen_outliers",
           local=True)

    fn.run(params={"num_boost_round": 40,
                   "verbose_eval": False,
                   "XGB_max_depth": 2,
                   "XGB_subsample": 0.9,
                   "test_set_key": "./artifacts/inputs/test-set"},
           inputs={"dataset": "./artifacts/inputs/xgb_custom.parquet"},
           handler="fit",
           local=True)

    assert(os.path.exists(os.getcwd() + "/plots/learning-curves.html"))
    delete_outputs({ARTIFACT_PATH, FUNCTION_PATH, PLOTS_PATH, RUNS_PATH, SCHEDULES_PATH})


def test_hub_xgb_custom():
    set_mlrun_hub_url(function_name="xgb_custom")
    fn = import_function("hub://xgb_custom")
    fn.run(params={"nrows": 8192,
                   "label_type": "float",
                   "local_path": "./artifacts/inputs/xgb_custom"},
           handler="gen_outliers",
           local=True)

    fn.run(params={"num_boost_round": 40,
                   "verbose_eval": False,
                   "XGB_max_depth": 2,
                   "XGB_subsample": 0.9,
                   "test_set_key": "./artifacts/inputs/test-set"},
           inputs={"dataset": "./artifacts/inputs/xgb_custom.parquet"},
           handler="fit",
           local=True)

    assert (os.path.exists(os.getcwd() + "/plots/learning-curves.html"))
    delete_outputs({ARTIFACT_PATH, FUNCTION_PATH, PLOTS_PATH, RUNS_PATH, SCHEDULES_PATH})
