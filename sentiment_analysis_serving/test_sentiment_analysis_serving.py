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
import os
import wget
from mlrun import import_function
import os.path
from os import path
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
    model = model_path + '/model.pt'
    if not path.exists(model):
        download_pretrained_model(model_path)
    fn = import_function('function.yaml')
    fn.add_model('model1', model_path=model, class_name='SentimentClassifierServing')
    # create an emulator (mock server) from the function configuration)
    server = fn.to_mock_server()

    instances = ['I had a pleasure to work with such dedicated team. Looking forward to \
                 cooperate with each and every one of them again.']
    result = server.test("/v2/models/model1/infer", {"inputs": instances})
    assert result['outputs']['predictions'][0] == 2


def test_meta_data():
    model_path = os.path.join(os.path.abspath('./'), 'models')
    model = model_path + '/model.pt'
    if not path.exists(model):
        download_pretrained_model(model_path)
    fn = import_function('function.yaml')
    fn.add_model('model1', model_path=model, class_name='SentimentClassifierServing')
    # create an emulator (mock server) from the function configuration)
    server = fn.to_mock_server()

    instances = ['I had a pleasure to work with such dedicated team. Looking forward to \
                 cooperate with each and every one of them again.']
    metadata = ['I AM VERY IMPORTANT']
    result = server.test("/v2/models/model1/infer", {"inputs": instances, "meta_data": metadata})
    assert result['outputs']['predictions'][0] == 2
    assert result['outputs']['meta_data'] == metadata
