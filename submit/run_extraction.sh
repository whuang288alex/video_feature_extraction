#!/bin/bash

# have job exit if any command returns with non-zero exit status
set -e
ENVNAME=feature_extraction
ENVDIR=$ENVNAME

# get input files from staging directory
cp /staging/groups/li_group_biostats/feature_extraction.tar.gz ./
cp /staging/groups/li_group_biostats/code.tar.gz ./

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# extract codes from the tar file
tar -xzf code.tar.gz

# run the extraction
python slurm.py --config-name $1

rm code.tar.gz feature_extraction.tar.gz
mkdir code
mv ./*.py ./code