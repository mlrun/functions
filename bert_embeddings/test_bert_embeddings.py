from bert_embeddings import init_context,handler
import nuclio
import json
import pickle
from functions.cli.helpers import delete_outputs
import numpy as np

FUNCTION_PATH = "functions"
SCHEDULE_PATH = "schedules"
RUNS_PATH = "runs"


def test_local_bert_embeddings():
    event = nuclio.Event(body=json.dumps(['John loves Mary']))
    ctx = nuclio.Context()
    init_context(ctx)
    outputs = pickle.loads(handler(ctx, event))
    assert(True if abs(np.mean(outputs[0]) - -0.011996539) <= 0.0001 else False) is True
    assert (True if abs(np.mean(outputs[0]) - -0.011996539) > 0 else False) is True
    delete_outputs({FUNCTION_PATH, SCHEDULE_PATH, RUNS_PATH})
