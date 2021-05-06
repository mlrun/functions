from describe import summarize
from mlrun import MLClientCtx

DATA_URL = 'https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv'


def test_summer():
    ctx = MLClientCtx()
    summarize(MLClientCtx, table=DATA_URL, label_column='label',update_dataset=True)

