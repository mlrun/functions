from transformers import BertModel, BertTokenizer
import torch
from typing import Union, List
import json
import pickle


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