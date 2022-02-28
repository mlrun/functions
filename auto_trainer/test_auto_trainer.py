import os

# List of servers:
import mlrun

SERVERS = [
    #            Server's suffix            |   User name   |             Access key               |    Index
    # ______________________________________|_______________|______________________________________|_____________
    ("yh38.iguazio-cd2.com",                 "admin",        "a75e2980-a003-420a-b14a-458e7fc8907a"),  # 0
    ("yh41.iguazio-cd1.com",                 "yonatan",        "3be37c76-6c7d-4472-8ec2-96cddcc6b010"),  # 1
]

# Chosen server:
SERVER_INDEX = 1

# Setup environment to use MLRun in the chosen server:
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["MLRUN_DBPATH"] = "https://mlrun-api.default-tenant.app.{}".format(
    SERVERS[SERVER_INDEX][0]
)  # MUST
os.environ["MLRUN_ARTIFACT_PATH"] = "/User/artifacts/{{project}}/{{run.uid}}"
os.environ["V3IO_USERNAME"] = SERVERS[SERVER_INDEX][1]  # MUST
os.environ["V3IO_API"] = "https://webapi.default-tenant.app.{}".format(
    SERVERS[SERVER_INDEX][0]
)  # MUST
os.environ["V3IO_ACCESS_KEY"] = SERVERS[SERVER_INDEX][2]  # MUST

import pandas as pd
from mlrun import import_function
from sklearn.svm import SVC
import pickle


DATASET_URL = "https://s3.wasabisys.com/iguazio/data/function-marketplace-data/xgb_trainer/classifier-data.csv"


def test_train():
    mlrun.get_or_create_project('auto-trainer', context="./", user_project=True)
    # Importing function:
    fn = import_function("function.yaml")

    # Creating model classes as inputs to the run:
    model_classes = {
        "sklearn.linear_model.LogisticRegression": {
            "CLASS_penalty": "l2",
            "CLASS_C": 0.1,
        },
        "xgboost.XGBRegressor": {
            "CLASS_max_depth": 3,
        },
        "lightgbm.LGBMClassifier": {
            "CLASS_max_depth": 3,
        },
    }

    try:
        for i, (model_class, kwargs) in enumerate(model_classes.items()):
            train_run = fn.run(
                inputs={"dataset": DATASET_URL},
                params={
                    "drop_columns": ["feat_0", "feat_2"],
                    "model_class": model_class,
                    "model_name": f"model_{i}",
                    # "tag": "",
                    "label_columns": "labels",
                    # "test_set": ''
                    "train_test_split_size": 0.2,
                    **kwargs,
                },
                handler="train",
                local=True,
            )
    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")


def test_evaluate():
    dataset = pd.read_csv(DATASET_URL)
    label = "labels"
    x, y = dataset.drop(labels=label, axis=1), dataset[label]
    model = SVC()
    model = model.fit(x, y)
    model_path = 'model.pkl'
    pickle.dump(model, open(model_path, 'wb'))

    # Importing function:
    fn = import_function("function.yaml")

    try:
        evaluate_run = fn.run(
            inputs={"dataset": DATASET_URL},
            params={
                "model": model_path,

                "drop_columns": ["feat_0", "feat_2"],
                "label_columns": "labels",
            },
            handler="evaluate",
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
    pass


def test_predict():
    pass
