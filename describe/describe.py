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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import PlotArtifact, TableArtifact

from yellowbrick.target import ClassBalance
from sklearn.preprocessing import StandardScaler

from typing import IO, AnyStr, Union, List, Optional

pd.set_option("display.float_format", lambda x: "%.2f" % x)

def _gcf_clear(plt):
    plt.cla()
    plt.clf()
    plt.close() 

def describe(
    context: MLClientCtx,
    table: Union[DataItem, str],
    label_column: str,
    class_labels: List[str],
    key: str = "table-summary",
) -> None:
    """Summarize a table

    TODO: merge with dask version

    :param context:         the function context
    :param table:           pandas dataframe
    :param key:             key of table summary in artifact store
    """
    _gcf_clear(plt)
    
    base_path = context.artifact_path
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path+"/plots", exist_ok=True)
    
    print(f'TABLE {table}')
    table = pd.read_parquet(str(table))
    header = table.columns.values
    
    # describe table
    sumtbl = table.describe()
    sumtbl = sumtbl.append(len(table.index)-table.count(), ignore_index=True)
    sumtbl.insert(0, "metric", ["count", "mean", "std", "min","25%", "50%", "75%", "max", "nans"])
    
    sumtbl.to_csv(os.path.join(base_path, key+".csv"), index=False)
    context.log_artifact(key, local_path=key+".csv")

    # plot class balance, record relative class weight
    _gcf_clear(plt)
    
    labels = table.pop(label_column)
    class_balance_model = ClassBalance(labels=class_labels)
    class_balance_model.fit(labels)
    
    scale_pos_weight = class_balance_model.support_[0]/class_balance_model.support_[1]
    context.log_artifact("scale_pos_weight", f"{scale_pos_weight:0.2f}")

    class_balance_model.show(outpath=os.path.join(base_path, "plots/imbalance.png"))
    context.log_artifact(PlotArtifact("imbalance", body=plt.gcf()), local_path="plots/imbalance.html")
    
    # plot feature correlation
    _gcf_clear(plt)
    tblcorr = table.corr()
    ax = plt.axes()
    sns.heatmap(tblcorr, ax=ax, annot=False, cmap=plt.cm.Reds)
    ax.set_title("features correlation")
    plt.savefig(os.path.join(base_path, "plots/corr.png"))
    context.log_artifact(PlotArtifact("correlation",  body=plt.gcf()), local_path="plots/corr.html")
    
    # plot histogram
    _gcf_clear(plt)
    ax = plt.axes()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    figarray = table.hist(ax=ax, ylabelsize=5, xlabelsize=5)
    for row in figarray:
        for f in row:
            f.set_title("")
    ax.set_title("features histogram")
    plt.savefig(os.path.join(base_path, "plots/hist.png"))
    context.log_artifact(PlotArtifact("histograms",  body=plt.gcf()), local_path="plots/hist.html")
   
    _gcf_clear(plt)
