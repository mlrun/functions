import os
import wget
from mlrun import import_function
from os import path
from rnn_serving import *

DATASET = np.array([[6.9955170e-01, 6.9952875e-01, 2.7922913e-02, 2.7853036e-02,
                     6.9955170e-01, 7.0086759e-01, 7.0118028e-01, 7.0142627e-01,
                     2.7922913e-02, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 6.9998503e-01, 1.6527303e-03, 2.7853036e-02,
                     7.0000792e-01, 7.0085293e-01, 7.0118028e-01, 7.0203447e-01,
                     1.6527303e-03, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0025057e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0027345e-01, 7.0014298e-01, 7.0190376e-01, 7.0128226e-01,
                     1.6904050e-04, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0144778e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0147055e-01, 7.0178574e-01, 7.0236105e-01, 7.0295709e-01,
                     7.3906886e-03, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0324355e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0326620e-01, 7.0308524e-01, 7.0490342e-01, 7.0427048e-01,
                     2.4815742e-03, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0324355e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0191067e-01, 7.0173001e-01, 7.0354480e-01, 7.0291305e-01,
                     2.9976186e-03, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0324355e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0166123e-01, 7.0148063e-01, 7.0284635e-01, 7.0249581e-01,
                     2.7904075e-03, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0324355e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0133996e-01, 7.0143080e-01, 7.0297277e-01, 7.0250750e-01,
                     4.1491759e-04, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0324355e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0150572e-01, 7.0251614e-01, 7.0281982e-01, 7.0370042e-01,
                     2.1256472e-03, 0.0000000e+00, 0.0000000e+00],
                    [6.9955170e-01, 7.0324355e-01, 1.6904050e-04, 2.7853036e-02,
                     7.0272487e-01, 7.0258951e-01, 7.0429617e-01, 7.0376801e-01,
                     1.4207334e-03, 0.0000000e+00, 0.0000000e+00]]).reshape(1, 10, 11).tolist()


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
    server.test("/v2/models/model2/infer", {"inputs": DATASET})
    # should add assert
