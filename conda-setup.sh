#!/usr/bin/env bash
# for some reason it is not 
cd $HOME
conda config --add channels conda-forge
conda config --add channels anaconda
conda create --yes -n mlrun-v0.4.6 numpy pandas scipy scikit-learn matplotlib seaborn pytest kfp pyarrow
conda install --yes -n mlrun-v0.4.6 ipykernel
conda install --yes -n mlrun-v0.4.6 -c DistrictDataLabs yellowbrick # to deprecate

eval "$(conda shell.bash hook)"
conda activate mlrun-v0.4.6
python -m ipykernel install --user --name=mlrun-v0.4.6
python -m pip install git+https://github.com/mlrun/mlrun.git@development
python -m pip install git+https://github.com/yjb-ds/mlutils.git
git clone https://github.com/functions/functions.git functions-tst