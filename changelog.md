# changelog

## 23-03-2020

* moved images to mlrun 0.4.6
    - images tagged as 0.4.6 == last development commit, and version tags (e.g., 0.4.5)
    - (WIP) updates made to multiple functions, mostly logging artifacts
    - parquet file extension changed from .pqt to .parquet (could also have been .pq)
* [load_dataset]()
    - fixed classification datasets
    - wrote README
* [sklearn_classifier](): 
    - load model config from json file as option
    - label binarizer fixed, didn't have correct classes
    - adjust metrics calcs for binary and multiclass
* [test_classifier]():
    - adjust metrics calcs for binary and multiclass
