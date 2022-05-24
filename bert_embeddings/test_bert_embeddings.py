from bert_embeddings import init_context,handler
import nuclio
import json
import pickle
import numpy as np

ARCHIVE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
ARTIFACTS_PATH = 'artifacts'


def test_bert_embeddings():
    event = nuclio.Event(body=json.dumps(['John loves Mary']))
    ctx = nuclio.Context()
    init_context(ctx)
    outputs = pickle.loads(handler(ctx, event))
    assert (True if abs(np.mean(outputs[0]) - -0.011996539) <= 0.0001 else False) is True
    assert (True if abs(np.mean(outputs[0]) - -0.011996539) > 0 else False) is True

