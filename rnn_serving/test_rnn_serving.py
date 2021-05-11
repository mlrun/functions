import os
import wget
from mlrun import import_function
from os import path
import mlrun
from pygit2 import Repository
from rnn_serving import *


def download_pretrained_model(model_path):
    # Run this to download the pre-trained model to your `models` directory
    model_location = 'https://s3.wasabisys.com/iguazio/models/bert/bert_classifier_v1.h5'
    saved_models_directory = model_path
    # Create paths
    os.makedirs(saved_models_directory, exist_ok=1)
    model_filepath = os.path.join(saved_models_directory, os.path.basename(model_location))
    wget.download(model_location, model_filepath)


def test_rnn_serving():
    model_path = os.path.join(os.path.abspath('./'), 'models')
    model = model_path + '/bert_classifier_v1.h5'
    if not path.exists(model):
        download_pretrained_model(model_path)

    fn = import_function('function.yaml')
    fn.add_model('mymodel', model_path=model, class_name='RNN_Model_Serving')
    # create an emulator (mock server) from the function configuration)
    server = fn.to_mock_server()
    # should add assert