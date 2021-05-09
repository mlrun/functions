from mlrun import code_to_function
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
        "file_ext": "csv"}, local=True,artifact_path="./")


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
        "label_column": "labels"},
        local=True, inputs={"dataset": 'classifier-data.csv'})

def test_xgb_test():
    DATA_PATH = "classifier-data.csv"
    MODELS_PATH = os.getcwd() + "/models/model.pkl"

    xgb_trainer()
    fn = code_to_function(name='test_xgb_test',
                          filename=os.path.dirname(os.path.dirname(__file__)) + "/xgb_test/xgb_test.py",
                          handler="xgb_test",
                          kind="job",
                          )
    fn.run(params={
        "label_column": "labels",
        "plots_dest": "plots/xgb_test"},
        local=True, inputs={"test_set": DATA_PATH,
                            "models_path": MODELS_PATH})
