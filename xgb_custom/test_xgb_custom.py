from mlrun import code_to_function
import os


def test_xgb_custom():
    fn = code_to_function(name='test_xgb_custom',
                          filename="xgb_custom.py",
                          handler="gen_outliers",
                          kind="job",
                          )
    outliers_run = fn.run(params= {'nrows': 8192, 'label_type': 'float'}, local=True)

    fn = code_to_function(name='test_fit_model',
                          filename="xgb_custom.py",
                          handler="fit",
                          kind="job",
                          )
    fit_run = fn.run(params={"num_boost_round" : 40,
                    "verbose_eval": False,
                    "XGB_max_depth": 2,
                    "XGB_subsample": 0.9},
                    inputs = {"dataset": 'xgb_custom.parquet'},
                    local=True)
