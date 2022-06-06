import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import mlrun

PRETRAINED_MODEL = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class BertSentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL)
        self.dropout = nn.Dropout(p=0.2)
        self.out_linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        out = self.dropout(pooled_out)
        out = self.out_linear(out)
        return self.softmax(out)


class SentimentClassifierServing(mlrun.serving.V2ModelServer):
    def load(self):
        """
        load bert model into class
        """
        model_file, _ = self.get_model('.pt')
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = BertSentimentClassifier(n_classes=3)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        self.model = model

    def predict(self, body):
        """
        predict function
        :param body: set of inputs for bert model to predict from
        """
        try:
            instances = body['inputs']
            enc = tokenizer.batch_encode_plus(instances, return_tensors='pt', pad_to_max_length=True)
            outputs = self.model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            _, predicts = torch.max(outputs, dim=1)
            prediction_return = {'predictions': predicts.cpu().tolist()}
            if "meta_data" in body:
                prediction_return['meta_data'] = body['meta_data']
            print("printing prediction return {}".format(prediction_return))
            return prediction_return
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
