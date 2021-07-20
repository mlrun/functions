import mlrun
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlrun.artifacts import PlotArtifact, TableArtifact
from mlrun.mlutils.plots import gcf_clear
import numpy as np


pd.set_option("display.float_format", lambda x: "%.2f" % x)

def summarize(
    context,
    dask_key: str = "dask_key",
    dataset: mlrun.DataItem = None,
    label_column: str = "label",
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
        #table = dataset.as_df(df_module=dd)
        table = dataset.as_df()
    else:
        context.logger.info(f"only these datasets are available {client.datasets} in client {client}")
        raise Exception("dataset not found on dask cluster")
    df = table
    header = df.columns.values
    extra_data = {}

    try:
        gcf_clear(plt)
        snsplt = sns.pairplot(df, hue=label_column)  # , diag_kws={"bw": 1.5})
        extra_data["histograms"] = context.log_artifact(
            PlotArtifact("histograms", body=plt.gcf()),
            local_path=f"{plots_dest}/hist.html",
            db_key=False,
        )
    except Exception as e:
        context.logger.error(f"Failed to create pairplot histograms due to: {e}")

    try:
        gcf_clear(plt)
        plot_cols = 3
        plot_rows = int((len(header) - 1) / plot_cols) + 1
        fig, ax = plt.subplots(plot_rows, plot_cols, figsize=(15, 4))
        fig.tight_layout(pad=2.0)
        for i in range(plot_rows * plot_cols):
            if i < len(header):
                sns.violinplot(
                    x=df[header[i]],
                    ax=ax[int(i / plot_cols)][i % plot_cols],
                    orient="h",
                    width=0.7,
                    inner="quartile",
                )
            else:
                fig.delaxes(ax[int(i / plot_cols)][i % plot_cols])
            i += 1
        extra_data["violin"] = context.log_artifact(
            PlotArtifact("violin", body=plt.gcf(), title="Violin Plot"),
            local_path=f"{plots_dest}/violin.html",
            db_key=False,
        )
    except Exception as e:
        context.logger.warn(f"Failed to create violin distribution plots due to: {e}")

    if label_column:
        labels = df.pop(label_column)
        imbtable = labels.value_counts(normalize=True).sort_index()
        try:
            gcf_clear(plt)
            balancebar = imbtable.plot(kind="bar", title="class imbalance - labels")
            balancebar.set_xlabel("class")
            balancebar.set_ylabel("proportion of total")
            extra_data["imbalance"] = context.log_artifact(
                PlotArtifact("imbalance", body=plt.gcf()),
                local_path=f"{plots_dest}/imbalance.html",
            )
        except Exception as e:
            context.logger.warn(f"Failed to create class imbalance plot due to: {e}")
        context.log_artifact(
            TableArtifact(
                "imbalance-weights-vec", df=pd.DataFrame({"weights": imbtable})
            ),
            local_path=f"{plots_dest}/imbalance-weights-vec.csv",
            db_key=False,
        )

    tblcorr = df.corr()
    mask = np.zeros_like(tblcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    dfcorr = pd.DataFrame(data=tblcorr, columns=header, index=header)
    dfcorr = dfcorr[np.arange(dfcorr.shape[0])[:, None] > np.arange(dfcorr.shape[1])]
    context.log_artifact(
        TableArtifact("correlation-matrix", df=tblcorr, visible=True),
        local_path=f"{plots_dest}/correlation-matrix.csv",
        db_key=False,
    )

    try:
        gcf_clear(plt)
        ax = plt.axes()
        sns.heatmap(tblcorr, ax=ax, mask=mask, annot=False, cmap=plt.cm.Reds)
        ax.set_title("features correlation")
        extra_data["correlation"] = context.log_artifact(
            PlotArtifact("correlation", body=plt.gcf(), title="Correlation Matrix"),
            local_path=f"{plots_dest}/corr.html",
            db_key=False,
        )
    except Exception as e:
        context.logger.warn(f"Failed to create features correlation plot due to: {e}")

    gcf_clear(plt)
