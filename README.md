# function hub (wip)


This functions hub is intended to be a centralized location for open source contributions of function components.  These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, it is expected that contributors follow certain guidelines/protocols (please chip-in).

## data

**[arc_to_parquet](arc_to_parquet/arc_to_parquet.ipynb)**<br>
download remote archive files and save to parquet

**[gen_class_data](gen_class_data/gen_class_data.ipynb)**<br>
generate simulated classification data according to detailed specs.  Great for testing algorithms and metrics and whole pipelines.

**[load_datasets](load_datasets/load_datasets.ipynb)**<br>
download toy datasets from sklearn, tensorflow datasets, and other data external curated datasets.

**[open_archive](open_archive/open_archive.ipynb)**<br>
download a zip or tar archive and extract its contents into a folder (preserving the directory structure)

**[load_dask](load_dask/load_dask.ipynb)**<br>
define a dask cluster, load your parquet data into it<br>

## explore

**[describe](describe/describe.ipynb)**<br>
estimate a set of descriptive statistics on pipeline data

**[describe_dask](describe/describe.ipynb)**<br>
estimate a set of descriptive statistics on pipeline data that has been loaded into a dask cluster

## model

**[aggregate](aggregate/aggregate.ipynb)**<br>
rolling aggregations on time seriesA

**[feature_selection](feature_selection/feature_selection.ipynb)**<br>
feture selection using the scikit feature-selection module

**[sklearn classifier](sklearn_classifier/sklearn_classifier.ipynb)**<br>
train any sklearn class has that has a fit function, including estimators, tranformers, etc...

**[xgb_trainer](xgb_trainer/xgb_trainer.ipynb)**<br>
train any one of 5 xgboost model types (classifier, regressor,...)

## serve

**[tf1_serving](tf1_serving/tf1_serving.ipynb)**<br>
deploy a tensorflow 1.x server

**[tf2_serving](tf2_serving/tf2_serving.ipynb)**<br>
deploy a tensorflow 2.x server

**[xgb_serving](xgb_serving/xgb_serving.ipynb)**<br>
deploy any xgboost model

**[model_server](model_server/model_server.ipynb)**<br>
deploy an scikit-learn or almost any pickled model

## test

**[model_server_tester](model_server_tester/model_server_tester.ipynb)**<br>
deploy an scikit-learn or almost any pickled model

**[test_classifier](test_classifier/test_classifier.ipynb)**<br>
test a classifier's model against help-out or new data

