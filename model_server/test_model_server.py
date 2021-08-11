from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import mlrun
import os
import requests
import json

def gen_model():
    # Getting the data
    X,y = load_iris(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
    # transforming the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Getting the model and training it
    classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
    classifier.fit(X_train, y_train)
    # saving the model
    filename = os.getcwd()+'/model.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    return X_test,y_test

def test_remote_model_server():
    x,y = gen_model()
    my_class = ClassifierModel('iris',model_dir=os.getcwd())
    my_class.load()
    my_dict = {'instances':x.tolist()}
    preds = my_class.predict(my_dict)
    assert(accuracy_score(y,preds) > 0.8)
