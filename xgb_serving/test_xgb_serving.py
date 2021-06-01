from mlrun import code_to_function, import_function
import os
from pygit2 import Repository
import mlrun
import pandas as pd


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
        "file_ext": "csv"}, local=True,artifact_path="./artifacts/inputs")


def xgb_trainer():
    get_class_data()
    fn = code_to_function(name='xgb_trainer',
                          filename=os.path.dirname(os.path.dirname(__file__)) + "/xgb_trainer/xgb_trainer.py",
                          handler="train_model",
                          kind="job",
                          )
    fn.run(params={
        "model_type": "classifier",
        "CLASS_tree_method": "hist",
        "CLASS_objective": "binary:logistic",
        "CLASS_booster": "gbtree",
        "FIT_verbose": 0,
        "label_column": "labels",
        "test_set": "./artifacts/test-set"},
        local=True, inputs={"dataset": './artifacts/inputs/classifier-data.csv'})


def set_mlrun_hub_url():
    xgb_trainer()
    branch = Repository('.').head.shorthand
    hub_url = "https://raw.githubusercontent.com/mlrun/functions/{}/xgb_serving/function.yaml".format(
        branch)
    mlrun.mlconf.hub_url = hub_url


# def test_xgb_serving():
#     model = os.getcwd() + "/models/model.pkl"
#     set_mlrun_hub_url()
#     fn = import_function('hub://xgb_serving')
#     fn.add_model('mymodel', model_path=model, class_name='XGBoostModel')
#     server = fn.to_mock_server()
#
#     # Testing the model
#     xtest = pd.read_csv('./artifacts/inputs/classifier-data.csv')
#     preds = server.predict({"instances": xtest.values[:10, :-1].tolist()})
#     assert(preds == [1, 0, 0, 0, 0, 0, 1, 1, 0, 1])