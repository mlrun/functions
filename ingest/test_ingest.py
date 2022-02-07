import os
import tempfile
import shutil
import datetime

import mlrun
import mlrun.feature_store as fstore
from mlrun.datastore.sources import CSVSource
from mlrun.feature_store.steps import *
from mlrun.features import MinMaxValidator
import pandas as pd


def _set_environment():
    artifact_path = tempfile.TemporaryDirectory().name
    os.makedirs(artifact_path)
    project = mlrun.new_project("ingest-test")
    return artifact_path, project


def _cleanup_environment(artifact_path: str):
    """
    Cleanup the test environment, deleting files and artifacts created during the test.

    :param artifact_path: The artifact path to delete.
    """
    # Clean the local directory:
    for test_output in [
        *os.listdir(artifact_path),
        "schedules",
        "runs",
        "artifacts",
        "functions",
    ]:
        test_output_path = os.path.abspath(f"./{test_output}")
        if os.path.exists(test_output_path):
            if os.path.isdir(test_output_path):
                shutil.rmtree(test_output_path)
            else:
                os.remove(test_output_path)

    # Clean the artifacts' directory:
    shutil.rmtree(artifact_path)


def create_dataframes() -> (pd.DataFrame, pd.DataFrame):
    quotes = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.030"),
                pd.Timestamp("2016-05-25 13:30:00.041"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.072"),
                pd.Timestamp("2016-05-25 13:30:00.075"),
            ],
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
        }
    )

    # move date:
    max_date = quotes["time"].max()
    now_date = datetime.datetime.now()
    delta = now_date - max_date
    quotes["time"] = quotes["time"] + delta

    return quotes


class MyMap(MapClass):
    def __init__(self, multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self._multiplier = multiplier

    def do(self, event):
        event["multi"] = event["bid"] * self._multiplier
        return event


def _create_feature_set():
    quotes_set = fstore.FeatureSet("stock-quotes", entities=[fstore.Entity("ticker")])

    quotes_set.graph.to("test_ingest.MyMap", multiplier=3).to(
        "storey.Extend", _fn="({'extra': event['bid'] * 77})"
    ).to("storey.Filter", "filter", _fn="(event['bid'] > 51.92)").to(
        FeaturesetValidator()
    )

    quotes_set.add_aggregation("ask", ["sum", "max"], "1h", "10m", name="asks1")
    quotes_set.add_aggregation("ask", ["sum", "max"], "5h", "10m", name="asks5")
    quotes_set.add_aggregation("bid", ["min", "max"], "1h", "10m", name="bids")

    # add feature validation policy
    quotes_set["bid"] = fstore.Feature(
        validator=MinMaxValidator(min=52, severity="info")
    )

    # add default target definitions
    quotes_set.set_targets()
    return quotes_set


def test_ingest():
    artifact_path, project = _set_environment()
    ingest_fn = mlrun.import_function("function.yaml")
    quotes = create_dataframes()

    quotes_set = _create_feature_set()
    quotes_set.save()

    data_uri = os.path.join(artifact_path, "quotes.csv")
    quotes.to_csv(data_uri, index=False)
    source = CSVSource("quotes", data_uri).to_dict()

    ingest_run = None
    try:
        ingest_run = ingest_fn.run(
            handler="ingest",
            params={
                "featureset": quotes_set.uri,
                "source": source,
            },
            local=True,
        )

    except Exception as exception:
        print(f"- The test failed - raised the following error:\n- {exception}")
    assert (
        fstore.get_feature_set(ingest_run.outputs["featureset"]).status.state
        == "created"
    ), "Targets not created successfully"
    _cleanup_environment(artifact_path)
