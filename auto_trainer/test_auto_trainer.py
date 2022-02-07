from mlrun import import_function

DATASET_URL = "https://s3.wasabisys.com/iguazio/data/function-marketplace-data/xgb_trainer/classifier-data.csv"


def test_train():
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
        }
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
