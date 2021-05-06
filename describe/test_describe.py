from describe import summarize
from mlrun import MLClientCtx
from mlrun.datastore import DataItem

DATA_URL = 'https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv'


def test_summarize():
    di = DataItem(url=DATA_URL)
    ctx = MLClientCtx()
    summarize(ctx, table=di, label_column='label', update_dataset=True)

