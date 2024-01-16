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
import pickle
import uuid
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from mlrun.frameworks.sklearn import apply_mlrun
from mlrun.projects import get_or_create_project
import mlrun
import mlrun.common.schemas
from batch_inference_v2 import infer
import shutil

REQUIRED_ENV_VARS = [
    "MLRUN_DBPATH",
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


def generate_data(n_samples: int = 5000, n_features: int = 20):
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


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_batch_predict():
    project = get_or_create_project(
        "batch-infer-test", context="./", user_project=True
    )

    # Configure test:
    n_samples = 5000
    n_features = 20

    # Create the function and run:
    test_function = mlrun.code_to_function(filename=__file__, kind="job")
    generate_data_run = test_function.run(
        handler="generate_data",
        params={"n_samples": n_samples, "n_features": n_features},
        returns=["training_set : dataset", "prediction_set : dataset"],
        local=True,
    )
    train_run = test_function.run(
        handler="train",
        inputs={"training_set": generate_data_run.outputs["training_set"]},
        local=True,
    )

    batch_predict_function = mlrun.import_function("function.yaml")
    batch_inference_run = batch_predict_function.run(
        handler="infer",
        inputs={"dataset": generate_data_run.outputs["prediction_set"]},
        params={
            "model_path": train_run.outputs["model"],
            "label_columns": "label",
            "trigger_monitoring_job": True,
            "perform_drift_analysis": True,
            "model_endpoint_drift_threshold": 0.2,
            "model_endpoint_possible_drift_threshold": 0.1,
        },
    )

    # Check the logged results:
    assert "batch_id" in batch_inference_run.status.results
    assert "drift_metric" in batch_inference_run.status.results
    assert batch_inference_run.status.results["drift_status"] is True

    # Check that 3 artifacts were generated
    assert len(batch_inference_run.status.artifacts) == 3

    # Check drift table artifact url
    assert (
        batch_inference_run.artifact("drift_table_plot").artifact_url
        == batch_inference_run.outputs["drift_table_plot"]
    )

    # Check the features drift results json:
    drift_results_file = batch_inference_run.artifact("features_drift_results").local()
    with open(drift_results_file, "r") as json_file:
        drift_results = json.load(json_file)
    assert len(drift_results) == n_features + 1

    # Clean resources
    _delete_project(project=project.metadata.name)


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
class TestBatchInferUnitTests:
    @classmethod
    def setup_class(cls):
        cls.project_name = "batch-infer-v2-unit-test"
        cls.infer_artifact_path = "./infer_test_result/"

    def setup_method(self):
        self.project = get_or_create_project(self.project_name)
        current_datetime = datetime.datetime.now()
        datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
        self.context = mlrun.get_or_create_ctx(datetime_str, project=self.project.metadata.name)
        self.context.artifact_path = self.infer_artifact_path

    def teardown_method(self):
        mlrun.get_run_db().delete_project(
            self.project.metadata.name,
            deletion_strategy=mlrun.common.schemas.DeletionStrategy.cascading,
        )
        if os.path.exists(self.infer_artifact_path):
            shutil.rmtree(self.infer_artifact_path)

    def test_infer(self):
        n_features = 10
        training_set, prediction_set = generate_data(n_features=n_features)
        clf = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic")
        x, y = prediction_set, training_set['target_label']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)
        clf.fit(x_train, y_train)
        train_set_to_log = x_train.join(y_train)
        model = self.project.log_model(f"clf_udbi{uuid.uuid4()}", body=pickle.dumps(clf),
                                       model_file=f"clf{uuid.uuid4()}.pkl", framework="xgboost",
                                       training_set=train_set_to_log, label_column="target_label")

        dataset = self.project.log_dataset(f"ds_lkr{uuid.uuid4()}", df=x_test)
        z_test = train_set_to_log * 5
        model_endpoint_sample_set = self.project.log_dataset(f"my_model_endpoint_sample_set{uuid.uuid4()}", df=z_test)


        list_model_endpoint_sample_set = generate_data(n_samples=4,n_features=n_features)[0].values.tolist()
        # model_endpoint_sample_set_value = model_endpoint_sample_set._df
        #model_endpoint_sample_set_value = model_endpoint_sample_set.uri
        model_endpoint_sample_set_value = list_model_endpoint_sample_set
        infer(context=self.context,
              batch_image_job="tomermamia855/mlrun-api:batch-infer-bug",
              dataset=dataset.to_dataitem().as_df(), model_path=model.uri,
              model_endpoint_sample_set= model_endpoint_sample_set_value,
              feature_columns=list(model_endpoint_sample_set.to_dataitem().as_df().columns),
              label_columns="target_label",
              log_result_set = False,
              model_endpoint_name=f"new-model-endpoint-sample_set{uuid.uuid4()}", artifacts_tag="tag1",
              trigger_monitoring_job=True, perform_drift_analysis=True, model_endpoint_drift_threshold=0.7,
              model_endpoint_possible_drift_threshold=0.5)


def _delete_project(project: str):
    mlrun.get_run_db().delete_project(
        project,
        deletion_strategy=mlrun.common.schemas.DeletionStrategy.cascading,
    )
