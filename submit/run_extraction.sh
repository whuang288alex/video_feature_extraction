#!/bin/bash

# have job exit if any command returns with non-zero exit status
set -e
ENVNAME=feature_extraction
ENVDIR=$ENVNAME

# get input files from staging directory
cp /staging/groups/li_group_biostats/feature_extraction.tar.gz ./
cp /staging/groups/li_group_biostats/code.tar.gz ./

# extract and set up the environment
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate
rm feature_extraction.tar.gz

# extract codes from the tar file
tar -xzf code.tar.gz
rm code.tar.gz

######################################
#  extract inputs from the tar file  #
#  TODO: modify this part if needed  #
######################################
cp /staging/groups/li_group_biostats/val_1.tar.gz ./
tar -xzf val_1.tar.gz
rm  val_1.tar.gz
mv val_1 videos

# run the extraction
python slurm.py --config-name $1

# remove other files to prevent them from being transferred back
rm ./*.py


tar -zcvf val_10.tar.gz ./*.pt
