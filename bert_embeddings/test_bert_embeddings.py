from bert_embeddings import init_context,handler
import nuclio
import json
import pickle

ARCHIVE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
ARTIFACTS_PATH = 'artifacts'


def test_bert_embeddings():
    event = nuclio.Event(body=json.dumps(['John loves Mary']))
    ctx = nuclio.Context()
    init_context(ctx)
    outputs = pickle.loads(handler(ctx, event))
    print(outputs)

