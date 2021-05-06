import mlrun
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
from mlrun.artifacts import PlotArtifact, TableArtifact
from mlrun.mlutils.plots import gcf_clear
from yellowbrick import ClassBalance
from typing import List

pd.set_option("display.float_format", lambda x: "%.2f" % x)

def summarize(
    context,
    dask_key: str = "dask_key",
    dataset: mlrun.DataItem = None,
    label_column: str = "label",
    class_labels: List[str] = [],
    plot_hist: bool = True,
    plots_dest: str = "plots",
    dask_function: str = None,
    dask_client=None,
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
    :param dask_function:   dask function url (db://..)
    :param dask_client:     dask client object
    """
    if dask_function:
        client = mlrun.import_function(dask_function).client
    elif dask_client:
        client = dask_client
    else:
        raise ValueError('dask client was not provided')
        
    if dask_key in client.datasets:
        table = client.get_dataset(dask_key)
    elif dataset:
        table = dataset.as_df(df_module=dd)
    else:
        context.logger.info(f"only these datasets are available {client.datasets} in client {client}")
        raise Exception("dataset not found on dask cluster")
    header = table.columns.values
    
    gcf_clear(plt)
    table = table.compute()
    snsplt = sns.pairplot(table, hue=label_column, diag_kws={'bw': 1.5})
    context.log_artifact(PlotArtifact('histograms',  body=plt.gcf()), 
                         local_path=f"{plots_dest}/hist.html")

    gcf_clear(plt)   
    labels = table.pop(label_column)
    if not class_labels:
        class_labels = labels.unique()
    class_balance_model = ClassBalance(labels=class_labels)
    class_balance_model.fit(labels)   
    scale_pos_weight = class_balance_model.support_[0]/class_balance_model.support_[1]
    context.log_result("scale_pos_weight", f"{scale_pos_weight:0.2f}")
    context.log_artifact(PlotArtifact("imbalance", body=plt.gcf()), 
                         local_path=f"{plots_dest}/imbalance.html")
    
    gcf_clear(plt)
    tblcorr = table.corr()
    ax = plt.axes()
    sns.heatmap(tblcorr, ax=ax, annot=False, cmap=plt.cm.Reds)
    ax.set_title("features correlation")
    context.log_artifact(PlotArtifact("correlation",  body=plt.gcf()), 
                         local_path=f"{plots_dest}/corr.html")
    gcf_clear(plt)

