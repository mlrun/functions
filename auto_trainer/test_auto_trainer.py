from mlrun import import_function

DATASET_URL = "https://s3.wasabisys.com/iguazio/data/function-marketplace-data/xgb_trainer/classifier-data.csv"


def test_train():
    fn = import_function("function.yaml")

    model_classes = {
        'sklearn.linear_model.LogisticRegression': {
            "MODEL_CLASS_penalty": 'l2',
            "MODEL_CLASS_C": 0.1,
        }
    }
    try:
        for model_class, kwargs in model_classes.items():
            train_run = fn.run(
                inputs={"dataset": DATASET_URL},
                params={
                    "drop_columns": ['feat_0', 'feat_2'],
                    "model_class": model_class,
                    "model_name": 'my_model',
                    # "tag": "",
                    "label_columns": "labels",
                    # "test_set": ''
                    "train_test_split_size": 0.2,
                    # "artifacts": {},
                    "kwargs": kwargs
                },
                handler="train",
                local=True,
            )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")

