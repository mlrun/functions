import mlrun.feature_store as fs
import pandas as pd


def create_dataframes():
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
    return quotes, trades


def test_get_offline_vector():
    quotes, trades = create_dataframes()

    vector = fs.FeatureVector("myvector", features, "stock-quotes.xx")
    resp = fs.get_offline_features(
        vector, entity_rows=trades, entity_timestamp_column="time", engine=engine
    )
    assert len(vector.spec.features) == len(
        features
    ), "unexpected num of requested features"
    assert (
        len(vector.status.features) == features_size
    ), "unexpected num of returned features"
    assert len(vector.status.stats) == features_size, "unexpected num of feature stats"
    assert vector.status.label_column == "xx", "unexpected label_column name"

    df = resp.to_dataframe()
    columns = trades.shape[1] + features_size - 2  # - 2 keys
    assert df.shape[1] == columns, "unexpected num of returned df columns"
    resp.to_parquet(str(self.results_path / f"query-{engine}.parquet"))

    # check simple api without join with other df
    # test the use of vector uri
    vector.save()
    resp = fs.get_offline_features(vector.uri, engine=engine)
    df = resp.to_dataframe()
    assert df.shape[1] == features_size, "unexpected num of returned df columns"
