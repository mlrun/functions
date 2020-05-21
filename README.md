# Functions hub 

This functions hub is intended to be a centralized location for open source contributions of function components.  
These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, 
it is expected that contributors follow certain guidelines/protocols (please chip-in).

## Functions


### [aggregate (job)](aggregate/aggregate.ipynb)

Rolling aggregation over Metrics and Lables according to specifications

categories: data-prep

### [arc-to-parquet (job)](arc_to_parquet/arc_to_parquet.ipynb)

retrieve remote archive, open and save as parquet

categories: data-movement, utils

### [churn-test (nuclio)](churn_server/churn_server.ipynb)

churn classification and predictor

categories: serving, ml

### [cox-test (job)](coxph_test/coxph_test.ipynb)

test a classifier using held-out or new data

categories: ml, test

### [cox-hazards (job)](coxph_trainer/coxph_trainer.ipynb)

train any classifier using scikit-learn's API

categories: training, ml

### [describe (job)](describe/describe.ipynb)

describe and visualizes dataset stats

categories: analysis

### [describe-dask (job)](describe_dask/describe_dask.ipynb)

describe and visualizes dataset stats

categories: analysis

### [feature-selection (job)](feature_selection/feature_selection.ipynb)

Select features through multiple Statistical and Model filters

categories: data-prep, ml

### [gen-class-data (job)](gen_class_data/gen_class_data.ipynb)

simulate classification data using scikit-learn

categories: simulators, ml

### [github-utils (job)](github_utils/github_utils.ipynb)

add comments to github pull requests

categories: notifications, utils

### [load-dask (dask)](load_dask/load_dask.ipynb)

load dask cluster with data

categories: data-movement, utils

### [load-dataset (job)](load_dataset/load_dataset.ipynb)

load a toy dataset from scikit-learn

categories: data-source, ml

### [sklearn-server (nuclio)](model_server/model_server.ipynb)

generic sklearn model server

categories: serving, ml

### [model-server-tester (job)](model_server_tester/model_server_tester.ipynb)

test model servers

categories: ml, test

### [open-archive (job)](open_archive/open_archive.ipynb)

Open a file/object archive into a target directory

categories: data-movement, utils

### [sklearn-classifier (job)](sklearn_classifier/sklearn_classifier.ipynb)

train any classifier using scikit-learn's API

categories: ml, training

### [slack-notify (job)](slack_notify/slack_notify.ipynb)

Send Slack notification

categories: ops

### [test-classifier (job)](test_classifier/test_classifier.ipynb)

test a classifier using held-out or new data

categories: ml, test

### [tensorflow-v1-2layers (nuclio)](tf1_serving/tf1_serving.ipynb)

tf1 image classification server

categories: serving, dl

### [tensorflow-v2-2layers (nuclio)](tf2_serving/tf2_serving.ipynb)

tf2 image classification server

categories: serving, dl

### [iris-xgb-serving (nuclio)](xgb_serving/xgb_serving.ipynb)

xgboost iris classification server

categories: serving, ml

### [xgb-test (job)](xgb_test/xgb_test.ipynb)

test a classifier using held-out or new data

categories: ml, test

### [xgb-trainer (job)](xgb_trainer/xgb_trainer.ipynb)

train multiple model types using xgboost

categories: training, ml, experimental
