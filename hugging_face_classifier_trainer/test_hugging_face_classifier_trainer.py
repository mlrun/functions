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
from mlrun import import_function

ADDITIONAL_PARAM_FOR_TRAIN = {
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


def test_train_sequence_classification():
    # Importing function:
    fn = import_function("function.yaml")

    train_run = None

    try:
        train_run = fn.run(
            params={
                "hf_dataset": "Shayanvsf/US_Airline_Sentiment",
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
                **ADDITIONAL_PARAM_FOR_TRAIN,
            },
            handler="train",
            local=True,
        )
    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
    assert train_run and all(
        key in train_run.outputs for key in ["model", "loss"]
    ), "outputs should include more data"


def test_train_and_optimize_sequence_classification():

    # Importing function:
    fn = import_function("function.yaml")

    train_run = None
    optimize_run = None

    try:
        train_run = fn.run(
            params={
                "hf_dataset": "Shayanvsf/US_Airline_Sentiment",
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
                **ADDITIONAL_PARAM_FOR_TRAIN,
            },
            handler="train",
            local=True,
        )

        optimize_run = fn.run(
            params={"model_path": train_run.outputs["model"]},
            handler="optimize",
            local=True,
        )
    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
    assert train_run and all(
        key in train_run.outputs for key in ["model", "loss"]
    ), "outputs should include more data"
    assert optimize_run and all(
        key in optimize_run.outputs for key in ["model"]
    ), "outputs should include more data"
