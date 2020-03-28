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

**[parquet_to_dask]()**<br>
define a dask cluster, load your parquet data into it<br>
access the dask client and dask dashboard throughout your mlrun pipeline<br>
combine it with other mlrun dask components to build distributed machine and deep learning pipelines

### modeling

**[feature engineering]()**

**[sklearn models]()**
there are literally undreds of functions available that conform to the **[Scikit Learn]()** API. 
this component enables training of any sklearn class has that has a fit function, so this includes estimators, tranformers, etc...<br>
sklearn classes are input as strings<br>
classes not in the sklearn package muct have an accompanying json file that is easy to create<br>
xgboost and lightgbm default model configs are included in the samples-configs folder<br>

**[xgboost/lightgbm]()**

**[neural networks]()**

**[tuning]()**


# Setup

The following setup instructions are for developers, and for regular users it is assumed that alot of this will be done by an as of **yet unwritten makefile script** that is part of the install process

1. for convenience, define some mlrun config path settings through environment variables
2. **[set up](#setup)** a minimal conda environment for reproducibility
3. do the **[suggested steps through functions](#suggetsed)** above

## set up a conda environment

**dont't forget to always select the correct environment for your notebooks**

run the script **[conda-setup](conda-setup)**

## GPU setup

many function components can take advantage of performance gains made available by running in a well-configured GPU environment.  Setting up the platform (client) side requires a custom conda environment like the one built above, with a few additions:

1. ensure drivers are available (cuda and cudnn, currently at 10.1 and 7)
2. add gpu compatible packages and drop-in replacements (e.g. tensorflow, pytorch,)
3. add nvidia's rapids library (which provide dask updates for gpu)
4. init conda environment scripts so libcuda can be found in addition to other libraries inside the environment (`/User/.conda/envs/<my-env>/etc/conda/activate.d/env_vars.sh` and 
`deactivate.d/env_vars.sh`)

all of this is done automatically in the conda-setup script **[conda-setup-gpu](conda-setup-gpu)**

