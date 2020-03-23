# changelog

## 23-03-2020

* generally:
    - moved images to 0.4.6
    - **(WIP failed) update log_artifact to log_dataset**
    - parquet file extension changed from .pqt to .parquet (could also have been .pq)
    - set `code_to_function` option `with_doc=True` to include function docs in yaml
    - ensure all artifacts saved in artifacts folder, gitignored
* **[load_dataset]()**
    - fixed classification datasets
    - wrote README
* **[sklearn_classifier]()** 
    - load model config from json file as option
    - label binarizer fixed, didn't have correct classes
    - adjust metrics calcs for binary and multiclass
    - fails on higgs with strange mlrun db http error
    - passes on wine, iris and breast_cancer
* **[test_classifier]()**
    - adjust metrics calcs for binary and multiclass
