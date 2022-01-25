import os
import tempfile
import shutil
import datetime

import pytest
import mlrun
import mlrun.feature_store as fstore
from mlrun.datastore.targets import CSVTarget
from mlrun.feature_store.steps import *
from mlrun.features import MinMaxValidator
from mlrun.run import get_dataitem


REQUIRED_ENV_VARS = [
    "MLRUN_DBPATH",
    "MLRUN_ARTIFACT_PATH",
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


def _set_environment():
    """
    Creating project and temp dir for the project.
    """
    artifact_path = tempfile.TemporaryDirectory().name
    os.makedirs(artifact_path)
    project = mlrun.get_or_create_project(
        "get-offline-features-test", context="./", user_project=True
    )
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

    # Clean the artifacts directory:
    shutil.rmtree(artifact_path)


def create_dataframes() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Creates all the necessary DataFrames to the test.
    """

    def move_date(df, col):
        max_date = df[col].max()
        now_date = datetime.datetime.now()
        delta = now_date - max_date
        df[col] = df[col] + delta
        return df

    stocks = pd.DataFrame(
        {
            "ticker": ["MSFT", "GOOG", "AAPL"],
            "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
            "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
        }
    )

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

    trades = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.038"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
            ],
            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
            "quantity": [75, 155, 100, 100, 100],
        }
    )
    quotes = move_date(quotes, "time")
    trades = move_date(trades, "time")
    return quotes, trades, stocks


class MyMap(MapClass):
    def __init__(self, multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self._multiplier = multiplier

    def do(self, event):
        event["multi"] = event["bid"] * self._multiplier
        return event


def _create_feature_set():
    """
    Creating all the necessary FeatureSets for the test.
    """
    stocks_set = fstore.FeatureSet("stocks", entities=[fstore.Entity("ticker")])

    quotes_set = fstore.FeatureSet("stock-quotes", entities=[fstore.Entity("ticker")])

    quotes_set.graph.to("MyMap", multiplier=3).to(
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

    # add default target definitions and plot
    quotes_set.set_targets()
    return quotes_set, stocks_set


@pytest.mark.skipif(
    condition=not _validate_environment_variables(),
    reason="Project's environment variables are not set",
)
def test_get_offline_vector():
    # Creating project:
    artifact_path, project = _set_environment()

    # Importing the marketplace function:
    gof_fn = mlrun.import_function("function.yaml")

    # Creating the dataframes:
    quotes, trades, stocks = create_dataframes()

    # Defining features for the FeatureVector:
    features = [
        "stock-quotes.multi",
        "stock-quotes.asks5_sum_5h as total_ask",
        "stock-quotes.bids_min_1h",
        "stock-quotes.bids_max_1h",
        "stocks.*",
    ]

    # Creating the FeatureSets and ingesting them:
    quotes_set, stocks_set = _create_feature_set()
    fstore.ingest(stocks_set, stocks)
    fstore.ingest(quotes_set, quotes)

    # Saving the trades dataframe as a csv to use as entity_rows:
    trades_uri = os.path.join(artifact_path, "trades.csv")
    trades.to_csv(trades_uri, index=False)

    # Creating target for the FeatureVector:
    target_dict = CSVTarget(
        "mycsv", path=os.path.join(artifact_path, "my_csv.csv")
    ).to_dict()

    # Creating the FeatureVector and saving it:
    vector = fstore.FeatureVector("stocks-vec", features)
    vector.save()

    # Running the getting_offline_features function:
    gof_run = None
    try:
        gof_run = gof_fn.run(
            handler="get_offline_features",
            inputs={"entity_rows": trades_uri},
            params={
                "feature_vector": vector.uri,
                "target": target_dict,
                "entity_timestamp_column": "time",
            },
            local=True,
        )

    except Exception as e:
        print(f"- The test failed - raised the following error:\n- {e}")

    target_df = (
        get_dataitem(target_dict["path"]).as_df()
        if os.path.exists(target_dict["path"])
        else None
    )
    vector_df = get_dataitem(gof_run.outputs["return"]).as_df()

    # Asserting that the target and FeatureVector dataframes are the same:
    assert vector_df.equals(target_df), "Target and feature vector are not the same"
    _cleanup_environment(artifact_path)
