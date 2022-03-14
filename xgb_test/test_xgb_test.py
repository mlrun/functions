from mlrun import code_to_function, import_function
import os
import pandas as pd


def get_class_data():
    fn = import_function("hub://gen_class_data")
    fn.run(params={
        "n_samples": 10_000,
        "m_features": 5,
        "k_classes": 2,
        "header": None,
        "weight": [0.5, 0.5],
        "sk_params": {"n_informative": 2},
        "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")


def xgb_trainer():
    get_class_data()
    fn = code_to_function(name='xgb_trainer',
                          filename=os.path.dirname(os.path.dirname(__file__)) + "/xgb_trainer/xgb_trainer.py",
                          handler="train_model",
                          kind="job",
                          )
    fn.run(params= {
        "model_type": "classifier",
        "CLASS_tree_method": "hist",
        "CLASS_objective": "binary:logistic",
        "CLASS_booster": "gbtree",
        "FIT_verbose": 0,
        "label_column": "labels",
        "test_set": "./artifacts/test-set"},
        local=True, inputs={"dataset": './artifacts/inputs/classifier-data.csv'})


def test_xgb_test_code_to_function():
    xgb_trainer()
    fn = code_to_function(name='test_xgb_test',
                          filename=os.path.dirname(os.path.dirname(__file__)) + "/xgb_test/xgb_test.py",
                          handler="xgb_test",
                          kind="job",
                          )
    fn.run(params={
        "label_column": "labels",
        "plots_dest": "plots/xgb_test"},
        local=True, inputs={"test_set": "./artifacts/inputs/classifier-data.csv",
                            "models_path": os.getcwd() + "/models/model.pkl"})

    assert(os.path.exists(os.getcwd() + "/models/model.pkl"))


def test_local_xgb_test_import_local_function():
    # importing data preparation function (gen_class_data) locally
    fn = import_function("hub://gen_class_data")
    fn.run(params={
        "n_samples": 10_000,
        "m_features": 5,
        "k_classes": 2,
        "header": None,
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
