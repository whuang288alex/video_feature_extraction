#!/bin/bash

# have job exit if any command returns with non-zero exit status
set -e

# download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
export HOME=$PWD
sh Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
export PATH=$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# create environment with script
conda create --name feature_extraction python=3.9
conda activate feature_extraction
python -m pip install -r requirements.txt

# get input files from staging directory
cp /staging/groups/li_group_biostats/code.tar.gz ./

# extract codes from the tar file
tar -xzf code.tar.gz

# run the extraction
python slurm.py --config-name $1

rm code.tar.gz feature_extraction.tar.gz
mkdir code
mv ./*.py ./code