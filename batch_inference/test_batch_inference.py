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
import json
import os
import tempfile

import mlrun
import numpy as np
import pandas as pd
from mlrun.frameworks.sklearn import apply_mlrun
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier


@mlrun.handler(outputs=["training_set", "prediction_set"])
def generate_data(n_samples: int = 5000, n_features: int = 20, n_classes: int = 2):
    # Generate a classification data:
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2)

    # Split the data into a training set and a prediction set:
    x_train, x_prediction = x[: n_samples // 2], x[n_samples // 2 :]
    y_train = y[: n_samples // 2]

    # Randomly drift some features:
    x_prediction += np.random.uniform(
        low=2, high=4, size=x_train.shape
    ) * np.random.randint(low=0, high=2, size=x_train.shape[1], dtype=int)

    # Initialize dataframes:
    features = [f"feature_{i}" for i in range(n_features)]
    training_set = pd.DataFrame(data=x_train, columns=features)
    training_set.insert(
        loc=n_features, column="target_label", value=y_train, allow_duplicates=True
    )
    prediction_set = pd.DataFrame(data=x_prediction, columns=features)

    return training_set, prediction_set


@mlrun.handler()
def train(training_set: pd.DataFrame):
    # Get the data into x, y:
    labels = pd.DataFrame(training_set["target_label"])
    training_set.drop(columns=["target_label"], inplace=True)

    # Initialize a model:
    model = DecisionTreeClassifier()

    # Apply MLRun:
    apply_mlrun(model=model, model_name="model")

    # Train:
    model.fit(training_set, labels)


def test_batch_predict():
    # Configure test:
    n_samples = 5000
    n_features = 20

    # Create the function and run:
    test_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    generate_data_run = test_function.run(
        handler="generate_data",
        artifact_path=artifact_path.name,
        params={"n_samples": n_samples, "n_features": n_features},
        local=True,
    )
    train_run = test_function.run(
        handler="train",
        artifact_path=artifact_path.name,
        inputs={"training_set": generate_data_run.outputs["training_set"]},
        local=True,
    )

    batch_predict_function = mlrun.import_function("function.yaml")
    batch_predict_run = batch_predict_function.run(
        handler="infer",
        artifact_path=artifact_path.name,
        inputs={"dataset": generate_data_run.outputs["prediction_set"]},
        params={
            "model": train_run.outputs["model"],
            # "label_columns": "label",
            "result_set_name": "result_set",
        },
        local=True,
    )

    # Check the result set:
    result_set = batch_predict_run.artifact("result_set").as_df()
    assert result_set.shape == (n_samples // 2, n_features + 1)
    assert "target_label" in result_set.columns
    assert "batch_id" in batch_predict_run.status.results

    # Check the drift table plot:
    assert (
        os.path.basename(batch_predict_run.artifact("drift_table_plot").local())
        == "drift_table_plot.html"
    )

    # Check the features drift results json:
    drift_results_file = batch_predict_run.artifact("features_drift_results").local()
    with open(drift_results_file, "r") as json_file:
        drift_results = json.load(json_file)
    assert len(drift_results) == n_features + 1

    # Check the final analysis logged results:
    assert "drift_status" in batch_predict_run.status.results
    assert "drift_metric" in batch_predict_run.status.results

    # Clear outputs:
    artifact_path.cleanup()
