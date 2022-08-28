# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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

