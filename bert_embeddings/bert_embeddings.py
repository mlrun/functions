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
import json
import pickle

import torch
from transformers import BertModel, BertTokenizer


def init_context(context):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    setattr(context.user_data, "tokenizer", tokenizer)
    setattr(context.user_data, "model", model)


def handler(context, event):
    docs = json.loads(event.body)
    docs = [doc.lower() for doc in docs]
    docs = context.user_data.tokenizer.batch_encode_plus(
        docs, pad_to_max_length=True, return_tensors="pt"
    )

    with torch.no_grad():
        embeddings = context.user_data.model(**docs)
    embeddings = [embeddings[0].numpy(), embeddings[1].numpy()]
    return pickle.dumps(embeddings)
