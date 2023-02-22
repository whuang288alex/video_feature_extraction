#!/bin/bash

# have job exit if any command returns with non-zero exit status
set -e
ENVNAME=feature_extraction
ENVDIR=$ENVNAME

# get input files from staging directory
cp /staging/groups/li_group_biostats/feature_extraction.tar.gz ./
cp /staging/groups/li_group_biostats/code.tar.gz ./
cp /staging/groups/li_group_biostats/inputs.tar.gz ./

# extract and set up the environment
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# extract codes from the tar file
tar -xzf code.tar.gz

# extract inputs from the tar file
tar -xzf inputs.tar.gz

# run the extraction
python slurm.py --config-name $1

rm code.tar.gz feature_extraction.tar.gz inputs.tar.gz
mkdir code
rm ./*.py 
mv ./features/*/*.pt ./
