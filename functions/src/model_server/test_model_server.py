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
import pickle

from model_server import ClassifierModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def gen_model():
    # Getting the data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    # transforming the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Getting the model and training it
    classifier = LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto")
    classifier.fit(X_train, y_train)
    # saving the model
    filename = os.getcwd() + "/model.pkl"
    pickle.dump(classifier, open(filename, "wb"))
    return X_test, y_test


def test_remote_model_server():
    x, y = gen_model()
    my_class = ClassifierModel("iris", model_dir=os.getcwd())
    my_class.load()
    my_dict = {"instances": x.tolist()}
    preds = my_class.predict(my_dict)
    assert accuracy_score(y, preds) > 0.8
