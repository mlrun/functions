# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import tempfile

import lightgbm as lgb
import mlflow
import mlflow.environment_variables
import mlflow.xgboost
import pytest
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

import os
# os.environ["MLRUN_IGNORE_ENV_FILE"] = "True"  #TODO remove before push

import mlrun
import mlrun.launcher.local
#  Important:
#  unlike mlconf which resets back to default after each test run, the mlflow configurations
#  and env vars don't, so at the end of each test we need to redo anything we set in that test.
#  what we cover in these tests: logging "regular" runs with, experiment name, run id and context
#  name (last two using mlconf), failing run mid-way, and a run with no handler.
#  we also test here importing of runs, artifacts and models from a previous run.

# simple mlflow example of lgb logging
def lgb_run():
    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # enable auto logging
    mlflow.lightgbm.autolog()

    train_set = lgb.Dataset(X_train, label=y_train)

    with mlflow.start_run():
        # train model
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.1,
            "metric": "multi_logloss",
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "seed": 42,
        }
        # model and training data are being logged automatically
        model = lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=[train_set],
            valid_names=["train"],
        )

        # evaluate model
        y_proba = model.predict(X_test)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})


# simple mlflow example of xgb logging
def xgb_run():
    # prepare train and test data
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # enable auto logging
    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    with mlflow.start_run():
        # train model
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": 0.3,
            "eval_metric": "mlogloss",
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "seed": 42,
        }
        # model and training data are being logged automatically
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])
        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})


@pytest.mark.parametrize("handler", ["xgb_run", "lgb_run"])
def test_track_run_with_experiment_name(handler):
    """
    This test is for tracking a run logged by mlflow into mlrun while it's running using the experiment name.
    first activate the tracking option in mlconf, then we name the mlflow experiment,
    then we run some code that is being logged by mlflow using mlrun,
    and finally compare the mlrun we tracked with the original mlflow run using the validate func
    """
    # Enable general tracking
    mlrun.mlconf.external_platform_tracking.enabled = True
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(f"{handler}_test_track")
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Create a project for this tester:
        project = mlrun.get_or_create_project(name="default", context=test_directory)

        # Create a MLRun function using the tester source file (all the functions must be located in it):
        func = project.set_function(
            func=__file__,
            name=f"{handler}-test",
            kind="job",
            image="mlrun/mlrun",
            requirements=["mlflow"],
        )
        # mlflow creates a dir to log the run, this makes it in the tmpdir we create
        trainer_run = func.run(
            local=True,
            handler=handler,
            artifact_path=test_directory,
        )

        serving_func = project.set_function(
            func=os.path.abspath("function.yaml"),
            name=f"{handler}-server",
        )
        model_name = f"{handler}-model"
        # Add the model
        upper_handler = handler.replace("_", "-")
        model_path = test_directory + f"/{upper_handler}-test-{upper_handler}/0/model/"
        serving_func.add_model(
            model_name,
            class_name="MLFlowModelServer",
            model_path=model_path,
        )

        # Create a mock server
        server = serving_func.to_mock_server()

        # An example taken randomly
        result = server.test(f"/v2/models/{model_name}/predict", {"inputs": [[5.1, 3.5, 1.4, 0.2]]})
    print(result)
    assert result
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


