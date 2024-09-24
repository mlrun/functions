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
import mlrun
# from pygit2 import Repository


MODEL_PATH = os.path.join(os.path.abspath("./"), "models")
MODEL = MODEL_PATH + "model.pt"


# def set_mlrun_hub_url():
#     branch = Repository(".").head.shorthand
#     hub_url = "https://raw.githubusercontent.com/mlrun/functions/{}/churn_server/function.yaml".format(
#         branch
#     )
#     mlrun.mlconf.hub_url = hub_url


def download_pretrained_model(model_path):
    # Run this to download the pre-trained model to your `models` directory
    import os

    model_location = None
    saved_models_directory = model_path
    # Create paths
    os.makedirs(saved_models_directory, exist_ok=1)
    model_filepath = os.path.join(
        saved_models_directory, os.path.basename(model_location)
    )
    wget.download(model_location, model_filepath)


def test_local_churn_server():
    # set_mlrun_hub_url()
    # model_path = os.path.join(os.path.abspath("./"), "models")
    # model = model_path + "/model.pt"
    # if not path.exists(model):
    #     download_pretrained_model(model_path)
    # fn = import_function("hub://churn_server")
    # fn.add_model("mymodel", model_path=model, class_name="ChurnModel")
    # # create an emulator (mock server) from the function configuration)
    # server = fn.to_mock_server()
    #
    # instances = [
    #     "I had a pleasure to work with such dedicated team. Looking forward to \
    #              cooperate with each and every one of them again."
    # ]
    # result = server.test("/v2/models/mymodel/infer", {"instances": instances})
    # assert result[0] == 2
    print("we need to download churn model")
