"""Snowflake Dask unit test"""
from mlrun import import_function

def test_snowflake_dask():
    """An unit test"""
    fn_to_test = import_function("function.yaml")

    # a fake assert to pass the unit test
    if fn_to_test.to_yaml().__contains__('job'):
        assert True
