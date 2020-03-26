#!/usr/bin/env bash

conda config --add channels conda-forge
conda config --add channels anaconda
conda create -n stable numpy pandas scipy scikit-learn matplotlib seaborn pytest kfp pyarrow
conda install -n stable ipykernel
conda install -n stable -c DistrictDataLabs yellowbrick # to deprecate

eval "$(conda shell.bash hook)"
conda activate stable
python -m ipykernel install --user --name=stable
python -m pip install git+https://github.com/mlrun/mlrun.git@development
python -m pip install git+https://github.com/yjb-ds/mlutils.git
git clone https://github.com/functions/functions.git@development functions-tst