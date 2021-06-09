from mlrun import import_function
from functions.cli.helpers import delete_outputs,set_mlrun_hub_url
import os
import pandas as pd


ARTIFACT_PATH = "artifacts"
FUNCTION_PATH = "functions"
MODELS_PATH = "models"
PLOTS_PATH = "plots"
RUNS_PATH = "runs"
SCHEDULES_PATH = "schedules"


def test_local_xgb_test():
    # importing data preparation function (gen_class_data) locally
    fn = import_function("../gen_class_data/function.yaml")
    fn.run(params={
        "n_samples": 10_000,
        "m_features": 5,
        "k_classes": 2,
        "weight": [0.5, 0.5],
        "sk_params": {"n_informative": 2},
        "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")

    # importing model training function (xgb_trainer) locally
    fn = import_function("../xgb_trainer/function.yaml")
    fn.run(params={
        "model_type": "classifier",
        "CLASS_tree_method": "hist",
        "CLASS_objective": "binary:logistic",
        "CLASS_booster": "gbtree",
        "FIT_verbose": 0,
        "label_column": "labels",
        "test_set": "./artifacts/test-set"},
        local=True, inputs={"dataset": './artifacts/inputs/classifier-data.csv'})

    # importing xgb_test function.yaml and running tests
    fn = import_function("function.yaml")
    fn.run(params={
        "label_column": "labels",
        "plots_dest": "plots/xgb_test"},
        local=True, inputs={"test_set": "./artifacts/inputs/classifier-data.csv",
                            "models_path": os.getcwd() + "/models/model.pkl"})

    # tests for gen_class_data
    assert (os.path.exists("./artifacts/inputs/classifier-data.csv")) is True
    df = pd.read_csv("artifacts/inputs/classifier-data.csv")
    assert (True if df["labels"].sum() == 5008 else False) is True
    # tests for xgb_trainer
    assert (os.path.exists(os.getcwd() + "/models/model.pkl"))
    # no tests for xgb_test (it is a test already)
    delete_outputs({ARTIFACT_PATH,FUNCTION_PATH,MODELS_PATH,PLOTS_PATH,RUNS_PATH,SCHEDULES_PATH})


def test_hub_xgb_test():
    # importing data preparation function (gen_class_data) from the hub
    set_mlrun_hub_url(function_name="gen_class_data")
    fn = import_function("hub://gen_class_data")
    fn.run(params={
        "n_samples": 10_000,
        "m_features": 5,
        "k_classes": 2,
        "weight": [0.5, 0.5],
        "sk_params": {"n_informative": 2},
        "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")

    # importing model training function (xgb_trainer) from the hub
    set_mlrun_hub_url(function_name="xgb_trainer")
    fn = import_function("hub://xgb_trainer")
    fn.run(params={
        "model_type": "classifier",
        "CLASS_tree_method": "hist",
        "CLASS_objective": "binary:logistic",
        "CLASS_booster": "gbtree",
        "FIT_verbose": 0,
        "label_column": "labels",
        "test_set": "./artifacts/test-set"},
        local=True, inputs={"dataset": './artifacts/inputs/classifier-data.csv'})

    # importing xgb_test from the hub and running tests
    set_mlrun_hub_url(function_name="xgb_test")
    fn = import_function("hub://xgb_test")
    fn.run(params={
        "label_column": "labels",
        "plots_dest": "plots/xgb_test"},
        local=True, inputs={"test_set": "./artifacts/inputs/classifier-data.csv",
                            "models_path": os.getcwd() + "/models/model.pkl"})

    # tests for gen_class_data
    assert (os.path.exists("./artifacts/inputs/classifier-data.csv")) is True
    df = pd.read_csv("artifacts/inputs/classifier-data.csv")
    assert (True if df["labels"].sum() == 5008 else False) is True
    # tests for xgb_trainer
    assert (os.path.exists(os.getcwd() + "/models/model.pkl"))
    # no tests for xgb_test (it is a test already)
    delete_outputs({ARTIFACT_PATH,FUNCTION_PATH,MODELS_PATH,PLOTS_PATH,RUNS_PATH,SCHEDULES_PATH})




