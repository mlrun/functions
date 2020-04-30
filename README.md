# function hub (wip)


This functions hub is intended to be a centralized location for open source contributions of function components.  These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, it is expected that contributors follow certain guidelines/protocols (please chip-in).

## suggested steps through functions

### data

**[arc_to_parquet]()**<br>
download remote archive files and save to parquet

**[gen_class_data]()**<br>
generate simulated classification data according to detailed specs.  Great for testing algorithms and metrics and whole pipelines.

**[load_datasets]()**<br>
download toy datasets from sklearn, tensorflow datasets, and other data external curated datasets.

**[open_archive]()**<br>
download a zip or tar archive and extract its contents into a folder (preserving the directory structure)

**[load_dask]()**<br>
define a dask cluster, load your parquet data into it<br>
access the dask client and dask dashboard throughout your mlrun pipeline<br>
combine it with other mlrun dask components to build distributed machine and deep learning pipelines

### modeling

**[sklearn classifier]()**
there are literally undreds of functions available that conform to the **[Scikit Learn]()** API. 
this component enables training of any sklearn class has that has a fit function, so this includes estimators, tranformers, etc...<br>
sklearn classes are input as strings<br>
classes not in the sklearn package muct have an accompanying json file that is easy to create<br>
xgboost and lightgbm default model configs are included in the samples-configs folder<br>

