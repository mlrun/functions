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
import json
from pathlib import Path
import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import Client

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import ChartArtifact, TableArtifact, PlotArtifact

from typing import List, Optional

def table_summary(
    context: MLClientCtx,
    dask_key: str = "dask_key",
    label_column: str = "labels",
    class_labels: List[str] = [],
    plot_hist: bool = True,
    plots_dest: str = "plots"
    alt_scheduler: str = None
) -> None:
    """Summarize a table
    
    Connects to dask client through the function context, or through an optional
    user-supplied scheduler.

    :param context:         the function context
    :param dask_key:        key of dataframe in dask client "datasets" attribute
    :param label_column:    ground truth column label
    :param class_labels:    label for each class in tables and plots
    :param plot_hist:       (True) set this to False for large tables
    :param plots_dest:      destination folder of summary plots (relative to artifact_path)
    :param alt_scheduler:   (None) an alternative scheduler file to connect with
    """
    # no client, 
    if not context.dask_client or alt_scheduler:
        dask_client = Client(scheduler_file=str(alt_scheduler))
    elif context.dask_client:
        dask_client = Client(scheduler_file=str(context.dask_client))
    else:
        raise Exception("out of luck, no dask_client or scheduler file!")
        
    if dask_key in dask_client.datasets
        df = dask_client.get_dataset(dask_key)
    else:
        context.logger.info(f"only these datasets are available {dask_client.datasets} in client {dask_client}")
        raise Exception("dataset not found on dask cluster")

    _gcf_clear(plt)   
    labels = table.pop(label_column)
    if not class_labels:
        class_labels = labels.unique()
    class_balance_model = ClassBalance(labels=class_labels)
    class_balance_model.fit(labels)   
    scale_pos_weight = class_balance_model.support_[0]/class_balance_model.support_[1]
    context.log_result("scale_pos_weight", f"{scale_pos_weight:0.2f}")
    context.log_artifact(PlotArtifact("imbalance", body=plt.gcf()), local_path=f"{plots_dest}/imbalance.html")
    
    _gcf_clear(plt)
    tblcorr = table.corr()
    ax = plt.axes()
    sns.heatmap(tblcorr, ax=ax, annot=False, cmap=plt.cm.Reds)
    ax.set_title("features correlation")
    context.log_artifact(PlotArtifact("correlation",  body=plt.gcf()), local_path=f"{plots_dest}/corr.html")
    # otherwise shows last plot:
    _gcf_clear(plt)    