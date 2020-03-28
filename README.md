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
3. do the **[suggested steps through functions](#suggetsed)** below

## set up a conda environment

### long way

to install the environment named **stable**, run the following commands in a terminal:

    conda config --add channels conda-forge
    conda config --add channels anaconda
    conda create -n stable numpy pandas scipy scikit-learn matplotlib seaborn pytest kfp pyarrow
    conda install -n stable ipykernel
    conda install -n stable -c DistrictDataLabs yellowbrick # to deprecate
    
at this point you should exit the terminal and refresh browser, open a new terminal:

    conda activate stable
    python -m ipykernel install --user --name=stable
    python -m pip install git+https://github.com/mlrun/mlrun.git@development
    git clone https://github.com/functions/functions.git@development
    
    # TODO
    git clone https://github.com/functions/functions.git@development
    functions/create-conda.sh or makefile/ci 



**dont't forget to always select the correct environment for your notebooks**

### short way

run the script conda-setup.sh

Here is a **partial `conda list`** of the included packages:

    # packages in environment at /User/.conda/envs/stable: 

    # Name                    Version                   Build  Channel 
    arrow-cpp                 0.15.1           py37h7cd5009_5    anaconda 
    blas                      1.0                         mkl    anaconda 
    intel-openmp              2020.0                      166    anaconda 
    ipython                   7.13.0                   pypi_0    pypi 
    joblib                    0.14.1                     py_0    anaconda 
    lightgbm                  2.3.0            py37he6710b0_0    anaconda 
    matplotlib                3.1.3                    py37_0    anaconda 
    mkl                       2019.5                      281    anaconda 
    numpy                     1.18.1           py37h4f9e942_0    anaconda 
    pandas                    1.0.2            py37h0573a6f_0    anaconda 
    pip                       20.0.2                   py37_1    anaconda 
    pyarrow                   0.15.1           py37h0573a6f_0    anaconda 
    pytest                    5.4.1                    py37_0    anaconda 
    python                    3.7.6                h0371630_2    anaconda 
    pyzmq                     19.0.0                   pypi_0    pypi 
    scikit-learn              0.22.1           py37hd81dba3_0    anaconda 
    scipy                     1.4.1            py37h0b6359f_0    anaconda 
    seaborn                   0.10.0                     py_0    anaconda 
    sqlite                    3.31.1               h7b6447c_0    anaconda 
    xgboost                   1.0.2            py37h3340039_0    conda-forge
    
## GPU setup

many function components can take advantage performance gains made available by running in a well-configured GPU environment.  Setting up the platform (client) side requires a custom conda environment like the one built above, with a few additions:

1. ensure drivers are available (cuda and cudnn, currently at 10.1 and 7)
2. add gpu compatible packages and drop-in replacements (e.g. tensorflow, pytorch,)
3. build gpu versions of packages when straightforward (e.g., xgboost)
3. add nvidia's rapids library (which provide dask updates for gpu)
4. init conda environment scripts so libcuda found (eg., activate.d/set_vars.sh and 
deactivate.d/unset_vars.sh)

all of this is done automatically in the conda-setup script **[conda-setup-gpu](conda-setup-gpu)**

