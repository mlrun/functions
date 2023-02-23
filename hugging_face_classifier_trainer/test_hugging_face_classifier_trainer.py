# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from typing import Tuple

import mlrun
import pandas as pd
import pytest
from mlrun import import_function

MODELS = [
    ("sklearn.linear_model.LinearRegression", "regression"),
    ("sklearn.ensemble.RandomForestClassifier", "classification"),
    ("xgboost.XGBRegressor", "regression"),
    ("lightgbm.LGBMClassifier", "classification"),
]

REQUIRED_ENV_VARS = [
    "MLRUN_DBPATH",
    "MLRUN_ARTIFACT_PATH",
    "V3IO_USERNAME",
    "V3IO_API",
    "V3IO_ACCESS_KEY",
]


def _validate_environment_variables() -> bool:
    """
    Checks that all required Environment variables are set.
    """
    environment_keys = os.environ.keys()
    return all(key in environment_keys for key in REQUIRED_ENV_VARS)


def _set_environment(env_file=None):
    if env_file:
        mlrun.set_env_from_file(env_file)
    mlrun.get_or_create_project(
        "hugging-face-classifier-trainer-test", context="./", user_project=True
    )


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_train_sequence_classification():
    _set_environment()

    # Importing function:
    fn = import_function("function.yaml")

    train_run = None

    additional_parameters = {
        "TRAIN_output_dir": "finetuning-sentiment-model-3000-samples",
        "TRAIN_learning_rate": 2e-5,
        "TRAIN_per_device_train_batch_size": 16,
        "TRAIN_per_device_eval_batch_size": 16,
        "TRAIN_num_train_epochs": 2,
        "TRAIN_weight_decay": 0.01,
        "TRAIN_push_to_hub": False,
        "TRAIN_evaluation_strategy": "epoch",
        "TRAIN_eval_steps": 1,
        "TRAIN_logging_steps": 1,
        "CLASS_num_labels": 2,
    }

    try:
        train_run = fn.run(
            params={
                "dataset_name": "Shayanvsf/US_Airline_Sentiment",
                "drop_columns": [
                    "airline_sentiment_confidence",
                    "negativereason_confidence",
                ],
                "pretrained_tokenizer": "distilbert-base-uncased",
                "pretrained_model": "distilbert-base-uncased",
                "model_class": "transformers.AutoModelForSequenceClassification",
                "label_name": "airline_sentiment",
                "num_of_train_samples": 100,
                "metrics": ["accuracy", "f1"],
                "random_state": 42,
                **additional_parameters,
            },
            handler="train",
            local=True,
        )
    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
    assert train_run and all(
        key in train_run.outputs for key in ["model", "loss"]
    ), "outputs should include more data"
