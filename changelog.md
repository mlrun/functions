# changelog

## 23-03-2020

* moved images to mlrun 0.4.6
    - images tagged as latest == last commit, and version tags (e.g., 0.4.5)
    - updates made to multiple functions, mostly logging artifacts
* [load_dataset]()
    - fixed classification datasets
* [sklearn_classifier](): 
    - load model config from json file as option
    - label binarizer fixed, didn't have correct classes
