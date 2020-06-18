# Functions hub 

This functions hub is intended to be a centralized location for open source contributions of function components.  
These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, 
it is expected that contributors follow certain guidelines/protocols (please chip-in).

## Functions

| function | kind | description | categories |
| --- | --- | --- | --- |
| [aggregate](aggregate/aggregate.ipynb) | job | Rolling aggregation over Metrics and Lables according to specifications | data-prep |
| [arc-to-parquet](arc_to_parquet/arc_to_parquet.ipynb) | job | retrieve remote archive, open and save as parquet | data-movement, utils |
| [bert-embeddings](bert_embeddings/bert_embeddings.ipynb) | nuclio | Get BERT based embeddings for given text | NLP, BERT, embeddings |
| [churn-test](churn_server/churn_server.ipynb) | nuclio | churn classification and predictor | serving, ml |
| [cox-test](coxph_test/coxph_test.ipynb) | job | test a classifier using held-out or new data | ml, test |
| [cox-hazards](coxph_trainer/coxph_trainer.ipynb) | job | train any classifier using scikit-learn's API | training, ml |
| [describe](describe/describe.ipynb) | job | describe and visualizes dataset stats | analysis |
| [describe-dask](describe_dask/describe_dask.ipynb) | job | describe and visualizes dataset stats | analysis |
| [feature-perms](feature_perms/feature_perms.ipynb) | job | estimate feature importances using permutations | analysis |
| [feature-selection](feature_selection/feature_selection.ipynb) | job | Select features through multiple Statistical and Model filters | data-prep, ml |
| [gen-class-data](gen_class_data/gen_class_data.ipynb) | job | simulate classification data using scikit-learn | simulators, ml |
| [github-utils](github_utils/github_utils.ipynb) | job | add comments to github pull requests | notifications, utils |
| [load-dask](load_dask/load_dask.ipynb) | dask | load dask cluster with data | data-movement, utils |
| [load-dataset](load_dataset/load_dataset.ipynb) | job | load a toy dataset from scikit-learn | data-source, ml |
| [sklearn-server](model_server/model_server.ipynb) | nuclio | generic sklearn model server | serving, ml |
| [model-server-tester](model_server_tester/model_server_tester.ipynb) | job | test model servers | ml, test |
| [open-archive](open_archive/open_archive.ipynb) | job | Open a file/object archive into a target directory | data-movement, utils |
| [sentiment-analysis-server](sentiment_analysis_serving/bert_sentiment_analysis_serving.ipynb) | nuclio | BERT based sentiment classification model | serving, NLP, BERT, sentiment analysis |
| [sklearn-classifier](sklearn_classifier/sklearn_classifier.ipynb) | job | train any classifier using scikit-learn's API | ml, training |
| [slack-notify](slack_notify/slack_notify.ipynb) | job | Send Slack notification | ops |
| [spark-submit](spark_submit/spark_submit.ipynb) | job | runing spark submit job | data-movement, ml |
| [test-classifier](test_classifier/test_classifier.ipynb) | job | test a classifier using held-out or new data | ml, test |
| [tensorflow-v1-2layers](tf1_serving/tf1_serving.ipynb) | nuclio | tf1 image classification server | serving, dl |
| [tensorflow-v2-2layers](tf2_serving/tf2_serving.ipynb) | nuclio | tf2 image classification server | serving, dl |
| [iris-xgb-serving](xgb_serving/xgb_serving.ipynb) | nuclio | xgboost iris classification server | serving, ml |
| [xgb-test](xgb_test/xgb_test.ipynb) | job | test a classifier using held-out or new data | ml, test |
| [xgb-trainer](xgb_trainer/xgb_trainer.ipynb) | job | train multiple model types using xgboost | training, ml, experimental |
