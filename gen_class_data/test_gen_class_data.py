from mlrun import code_to_function,import_function
import os
from functions.cli.helpers import delete_outputs,set_mlrun_hub_url
import pandas as pd


ARTIFACT_PATH = "artifacts"
RUNS_PATH = "runs"
SCHEDULES_PATH = "schedules"


def test_local_gen_class_data():
    fn = import_function("function.yaml")
    fn.run(params={
        "n_samples": 10_000,
        "m_features": 5,
        "k_classes": 2,
        "weight": [0.5, 0.5],
        "sk_params": {"n_informative": 2},
        "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")

    df = pd.read_csv("artifacts/inputs/classifier-data.csv")
    assert (os.path.exists("./artifacts/inputs/classifier-data.csv")) is True
    assert (df["labels"].sum() == 5008) is True
    delete_outputs({RUNS_PATH,SCHEDULES_PATH,ARTIFACT_PATH})


def test_gen_class_data():
    set_mlrun_hub_url(function_name="gen_class_data")
    fn = import_function("hub://gen_class_data")
    fn.run(params={
            "n_samples": 10_000,
            "m_features": 5,
            "k_classes": 2,
            "weight": [0.5, 0.5],
            "sk_params": {"n_informative": 2},
            "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")

    assert(os.path.exists("./artifacts/inputs/classifier-data.csv")) is True
    df = pd.read_csv("artifacts/inputs/classifier-data.csv")
    assert(True if df["labels"].sum()== 5008 else False) is True
    delete_outputs({RUNS_PATH, SCHEDULES_PATH, ARTIFACT_PATH})

