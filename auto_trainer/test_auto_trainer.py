import mlrun
import pandas as pd
from mlrun import import_function

DATASET_URL = "https://s3.wasabisys.com/iguazio/data/function-marketplace-data/xgb_trainer/classifier-data.csv"


def test_train():
    df = pd.read_csv(DATASET_URL)
    fn = import_function("function.yaml")

    train_run = fn.run(
        params={
            "label_column": "labels",
            "model_class": 'xgboost.XGBRegressor'
        },
        inputs={"dataset": DATASET_URL},
        handler="train",
        local=True,
    )
