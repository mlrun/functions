from coxph_trainer import train_model
from mlrun import get_or_create_ctx
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from cloudpickle import dumps, dump, load
from sklearn.preprocessing import (OneHotEncoder,LabelEncoder)
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
import mlrun
import re
import subprocess
from pygit2 import Repository


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

def _set_mlrun_hub_url(repo_name = None, branch_name = None, function_name = None):
    repo_name =  re.search("\.com/.*?/", str(subprocess.run(['git', 'remote', '-v'], stdout=subprocess.PIPE).stdout)).group()[5:-1] if not repo_name else repo_name
    branch_name = Repository('.').head.shorthand if not branch_name else branch_name
    function_name = "" if not function_name else function_name # MUST ENTER FUNCTION NAME !!!!
    hub_url = f"https://raw.githubusercontent.com/{repo_name}/functions/{branch_name}/{function_name}/function.yaml"
    mlrun.mlconf.hub_url = hub_url

def test_local_coxph_train():
    ctx = get_or_create_ctx(name="tasks survive trainer")
    data_url = "https://raw.githubusercontent.com/mlrun/demos/0.6.x/customer-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    src = mlrun.get_dataitem(data_url)
    data_clean(context=ctx,src=src)
    dataset = pd.read_csv("encoded-data.csv")
    train_model(strata_cols=['InternetService', 'StreamingMovies', 'StreamingTV', 'PhoneService'],
                encode_cols={"Contract": "Contract", "PaymentMethod": "Payment"},
                models_dest='models/cox',
                dataset=dataset, context=ctx)

    model =load(open("models/cox/km/model.pkl", "rb"))
    ans = model.predict([1,10,30,100,200])

def test_remote_coxph_train():
    ctx = get_or_create_ctx(name="tasks survive trainer")
    data_url = "https://raw.githubusercontent.com/mlrun/demos/0.6.x/customer-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    src = mlrun.get_dataitem(data_url)
    data_clean(context=ctx, src=src)
    _set_mlrun_hub_url(function_name="coxph_trainer")
    fn = mlrun.import_function("hub://coxph_trainer")
    fn.run(params={"event_column": "labels",
                     "strata_cols": ['InternetService', 'StreamingMovies', 'StreamingTV', 'PhoneService'],
                     "encode_cols": {"Contract": "Contract", "PaymentMethod": "Payment"},
                     "models_dest": 'models/cox'},
           inputs={"dataset": "encoded-data.csv"}, local=True)

    model = load(open("models/cox/km/model.pkl", "rb"))
    ans = model.predict([1, 10, 30, 100, 200])