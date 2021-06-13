# Functions hub 

This functions hub is intended to be a centralized location for open source contributions of function components.  
These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, 
it is expected that contributors follow certain guidelines/protocols (please chip-in).

## Functions
| function | kind | description | categories |
| --- | --- | --- | --- |
| [aggregate](aggregate/aggregate.ipynb) | job | Rolling aggregation over Metrics and Lables according to specifications | data-prep |
| [bert-embeddings](bert_embeddings/bert_embeddings.ipynb) | nuclio | Get BERT based embeddings for given text | NLP, BERT, embeddings |
| [churn-server](churn_server/churn_server.ipynb) | nuclio | churn classification and predictor | serving, ml |
| [concept-drift](concept_drift/concept_drift.ipynb) | job | Deploy a streaming Concept Drift detector on a labeled stream | ml, serve |
| [concept-drift-streaming](concept_drift_streaming/concept_drift_streaming.ipynb) | nuclio | Deploy a streaming Concept Drift detector on a labeled stream. the nuclio part of the concept_drift function | ml, serve |
| [coxph-test](coxph_test/coxph_test.ipynb) | job | Test cox proportional hazards model | ml, test |
| [coxph-trainer](coxph_trainer/coxph_trainer.ipynb) | job | cox proportional hazards, kaplan meier plots | training, ml |
| [describe](describe/describe.ipynb) | job | describe and visualizes dataset stats | analysis |
| [describe-dask](describe_dask/describe_dask.ipynb) | job | describe and visualizes dataset stats | analysis |
| [describe-spark](describe_spark/describe_spark.ipynb) | job |  |  |
| [feature-perms](feature_perms/feature_perms.ipynb) | job | estimate feature importances using permutations | analysis |
| [feature-selection](feature_selection/feature_selection.ipynb) | job | Select features through multiple Statistical and Model filters | data-prep, ml |
| [gen-class-data](gen_class_data/gen_class_data.ipynb) | job | Create a binary classification sample dataset and save. | data-prep |
| [github-utils](github_utils/github_utils.ipynb) | job | add comments to github pull request | notifications, utils |
| [load-dask](load_dask/load_dask.ipynb) | dask | load dask cluster with data | data-movement, utils |
| [load-dataset](load_dataset/load_dataset.ipynb) | job | load a toy dataset from scikit-learn | data-source, ml |
| [model-monitoring-batch](model_monitoring_batch/model_monitoring_batch.ipynb) | job |  |  |
| [model-monitoring-stream](model_monitoring_stream/model_monitoring_stream.ipynb) | nuclio |  |  |
| [model-server](model_server/model_server.ipynb) | nuclio | generic sklearn model server | serving, ml |
| [model-server-tester](model_server_tester/model_server_tester.ipynb) | job | test model servers | ml, test |
| [pandas-profiling-report](pandas_profiling_report/pandas_profiling_report.ipynb) | job | Create Pandas Profiling Report from Dataset | analysis |
| [project-runner](project_runner/project_runner.ipynb) | nuclio | Nuclio based - Cron scheduler for running your MLRun projects | utils |
| [rnn-serving](rnn_serving/rnn_serving.ipynb) | serving | deploy an rnn based stock analysis model server. | model-serving |
| [send-email](send_email/send_email.ipynb) | job | Send Email messages through SMTP server | notifications |
| [sentiment-analysis-serving](sentiment_analysis_serving/sentiment_analysis_serving.ipynb) | serving | BERT based sentiment classification model | serving, NLP, BERT, sentiment analysis |
| [sklearn-classifier](sklearn_classifier/sklearn_classifier.ipynb) | job | train any classifier using scikit-learn's API | ml, training |
| [sklearn-classifier-dask](sklearn_classifier_dask/sklearn_classifier_dask.ipynb) | job | train any classifier using scikit-learn's API over Dask | ml, training, dask |
| [slack-notify](slack_notify/slack_notify.ipynb) | job | Send Slack notification | ops |
| [sql-to-file](sql_to_file/sql_to_file.ipynb) | job | SQL To File - Ingest data using SQL query | data-prep |
| [stream-to-parquet](stream_to_parquet/stream_to_parquet.ipynb) | nuclio | Saves a stream to Parquet and can lunch drift detection task on it | ml, serve |
| [test-classifier](test_classifier/test_classifier.ipynb) | job | test a classifier using held-out or new data | ml, test |
| [tf1-serving](tf1_serving/tf1_serving.ipynb) | nuclio | tf1 image classification server | serving, dl |
| [tf2-serving](tf2_serving/tf2_serving.ipynb) | nuclio | tf2 image classification server | serving, dl |
| [tf2-serving-v2](tf2_serving_v2/tf2_serving_v2.ipynb) | serving | tf2 image classification server v2 | serving, dl |
| [v2-model-server](v2_model_server/v2_model_server.ipynb) | serving | generic sklearn model server | serving, ml |
| [v2-model-tester](v2_model_tester/v2_model_tester.ipynb) | job | test v2 model servers | ml, test |
| [virtual-drift](virtual_drift/virtual_drift.ipynb) | job | Compute drift magnitude between Time-Samples T and U | ml, serve, concept-drift |
| [xgb-custom](xgb_custom/xgb_custom.ipynb) | job | simulate data with outliers. | model-testing |
| [xgb-serving](xgb_serving/xgb_serving.ipynb) | nuclio | xgboost test data classification server | model-serving |
| [xgb-test](xgb_test/xgb_test.ipynb) | job | Test one or more classifier models against held-out dataset. | model-test |
| [xgb-trainer](xgb_trainer/xgb_trainer.ipynb) | job | train multiple model types using xgboost. | model-prep |
