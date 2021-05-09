from mlrun import code_to_function, import_function
import os
from pygit2 import Repository
import mlrun


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
        "file_ext": "csv"}, local=True,artifact_path="./")


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
        "label_column": "labels"},
        local=True, inputs={"dataset": 'classifier-data.csv'})


def set_mlrun_hub_url():
    xgb_trainer()
    branch = Repository('.').head.shorthand
    hub_url = "https://raw.githubusercontent.com/mlrun/functions/{}/sentiment_analysis_serving/function.yaml".format(
        branch)
    hub_url = "https://raw.githubusercontent.com/daniels290813/functions/development/xgb_serving/function.yaml"
    mlrun.mlconf.hub_url = hub_url


def test_xgb_serving():
    # DATA_PATH = "classifier-data.csv"
    model = os.getcwd() + "/models/model.pkl"
    set_mlrun_hub_url()
    fn = import_function('hub://xgb_serving')
    fn.add_model('mymodel', model_path=model, class_name='XGBoostModel')
    # server = fn.to_mock_server()