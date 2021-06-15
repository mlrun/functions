from mlrun import code_to_function, import_function
import os


def get_class_data():
    fn = code_to_function(name='test_gen_class_data',
                          filename= os.path.dirname(os.path.dirname(__file__)) + "/gen_class_data/gen_class_data.py",
                          handler="gen_class_data",
                          kind="job",
                          )
    fn.run(params={
        "n_samples": 10_000,
        "m_features": 5,
        "k_classes": 2,
        "weight": [0.5, 0.5],
        "sk_params": {"n_informative": 2},
        "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")


def test_xgb_trainer_code_to_function():
    get_class_data()
    fn = code_to_function(name='test_xgb_trainer',
                          filename="xgb_trainer.py",
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

    assert(os.path.exists(os.getcwd() + "/models/model.pkl"))


def test_local_xgb_trainer_import_function():
    # importing data preparation function locally
    fn = import_function("../gen_class_data/function.yaml")
    fn.run(params={
        "n_samples": 10_000,
        "m_features": 5,
        "k_classes": 2,
        "weight": [0.5, 0.5],
        "sk_params": {"n_informative": 2},
        "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")

    fn = import_function("function.yaml")
    fn.run(params={
        "model_type": "classifier",
        "CLASS_tree_method": "hist",
        "CLASS_objective": "binary:logistic",
        "CLASS_booster": "gbtree",
        "FIT_verbose": 0,
        "label_column": "labels",
        "test_set": "./artifacts/test-set"},
        local=True, inputs={"dataset": './artifacts/inputs/classifier-data.csv'})

    assert (os.path.exists(os.getcwd() + "/models/model.pkl"))
