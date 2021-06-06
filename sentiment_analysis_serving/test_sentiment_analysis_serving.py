import os
import wget
from mlrun import import_function
import os.path
from os import path
import mlrun
from pygit2 import Repository
from sentiment_analysis_serving import *


MODEL_PATH = os.path.join(os.path.abspath('./'), 'models')
MODEL = MODEL_PATH + "model.pt"


def download_pretrained_model(model_path):
    # Run this to download the pre-trained model to your `models` directory
    import os
    model_location = 'https://iguazio-sample-data.s3.amazonaws.com/models/model.pt'
    saved_models_directory = model_path
    # Create paths
    os.makedirs(saved_models_directory, exist_ok=1)
    model_filepath = os.path.join(saved_models_directory, os.path.basename(model_location))
    wget.download(model_location, model_filepath)


def test_local_sentiment_analysis_serving():
    model_path = os.path.join(os.path.abspath('./'), 'models')
    model = model_path+'/model.pt'
    if not path.exists(model):
        download_pretrained_model(model_path)
    branch = Repository('.').head.shorthand
    fn = import_function('hub://sentiment_analysis_serving:{}'.format(branch))
    fn.add_model('model1', model_path=model, class_name='SentimentClassifierServing')
    # create an emulator (mock server) from the function configuration)
    server = fn.to_mock_server()

    instances = ['I had a pleasure to work with such dedicated team. Looking forward to \
                 cooperate with each and every one of them again.']
    result = server.test("/v2/models/model1/infer", {"inputs": instances})
    assert result['outputs'][0] == 2