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
from mlrun import get_or_create_ctx, import_function
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from cloudpickle import dumps, load
from sklearn.preprocessing import LabelEncoder
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
import mlrun

ARTIFACT_PATH = "artifacts"
FUNCTION_PATH = "functions"
MODELS_PATH = "models"
PLOTS_PATH = "plots"
RUNS_PATH = "runs"
SCHEDULES_PATH = "schedules"
DATA_URL = "https://raw.githubusercontent.com/mlrun/demos/0.6.x/customer-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def data_clean(
        context: MLClientCtx,
        src: DataItem,
        file_ext: str = "csv",
        models_dest: str = "models/encoders",
        cleaned_key: str = "cleaned-data",
        encoded_key: str = "encoded-data"
):
    df = src.as_df()

    # drop columns
    drop_cols_list = ["customerID", "TotalCharges"]
    df.drop(drop_cols_list, axis=1, inplace=True)

    # header transformations
    old_cols = df.columns
    rename_cols_map = {
        "SeniorCitizen": "senior",
        "Partner": "partner",
        "Dependents": "deps",
        "Churn": "labels"
    }
    df.rename(rename_cols_map, axis=1, inplace=True)

    # add drop column to logs:
    for col in drop_cols_list:
        rename_cols_map.update({col: "_DROPPED_"})

    # log the op
    tp = os.path.join(models_dest, "preproc-column_map.json")
    context.log_artifact("preproc-column_map.json",
                         body=json.dumps(rename_cols_map),
                         local_path=tp)
    df = df.applymap(lambda x: "No" if str(x).startswith("No ") else x)

    # encode numerical type as category bins (ordinal)
    bins = [0, 12, 24, 36, 48, 60, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    tenure = df.tenure.copy(deep=True)
    df["tenure_map"] = pd.cut(df.tenure, bins, labels=False)
    tenure_map = dict(zip(bins, labels))
    # save this transformation
    tp = os.path.join(models_dest, "preproc-numcat_map.json")
    context.log_artifact("preproc-numcat_map.json",
                         body=bytes(json.dumps(tenure_map).encode("utf-8")),
                         local_path=tp)

    context.log_dataset(cleaned_key, df=df, format=file_ext, index=False)
    fix_cols = ["gender", "partner", "deps", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "PhoneService",
                "MultipleLines", "PaperlessBilling", "InternetService",
                "Contract", "PaymentMethod", "labels"]

    d = defaultdict(LabelEncoder)
    df[fix_cols] = df[fix_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
    context.log_dataset(encoded_key, df=df, format=file_ext, index=False)

    model_bin = dumps(d)
    context.log_model("model",
                      body=model_bin,
                      artifact_path=os.path.join(context.artifact_path,
                                                 models_dest),
                      model_file="model.pkl")


def test_local_coxph_train():
    # ctx = get_or_create_ctx(name="tasks survive trainer")
    # src = mlrun.get_dataitem(DATA_URL)
    data_clean_function = mlrun.code_to_function(
        filename="test_coxph_trainer.py",
        name="data_clean",
        kind="job",
        image="mlrun/mlrun",
    )
    data_clean_run = data_clean_function.run(
        handler="data_clean",
        inputs={"src": DATA_URL},
        params={
            "cleaned_key": "cleaned-data",
            "encoded_key": "encoded-data",
        },
        local=True,
        artifact_path='./'
    )

    trainer_fn = import_function("function.yaml")
    trainer_run = trainer_fn.run(
        params={
            "strata_cols": ['InternetService', 'StreamingMovies', 'StreamingTV', 'PhoneService'],
            "encode_cols": {"Contract": "Contract", "PaymentMethod": "Payment"},
            "models_dest": 'models/cox'
        },
        inputs={"dataset": data_clean_run.artifact("encoded-data").url},
        local=True,
        artifact_path='./'
    )

    model = load(open(f"{trainer_run.artifact('km-model').url}model.pkl", "rb"))
    ans = model.predict([1, 10, 30, 100, 200])
    assert(sum([abs(x-y) for x, y in zip(list(np.around(ans, 2)), [0.95, 0.85, 0.77, 0.58, 0.58])]) < 0.5)