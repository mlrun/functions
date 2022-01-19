import mlrun.feature_store as fs


def test_get_offline_vector():

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
    assert (
            len(vector.status.stats) == features_size
    ), "unexpected num of feature stats"
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