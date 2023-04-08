#!/bin/bash

# TODO: modify this part if needed
ENVNAME=feature_extraction      
STAGING_DIR=groups/li_group_biostats
INPUT=THUMOS14_test

# basic set up
set -e
export PATH
mkdir $ENVNAME
tar -xzf /staging/$STAGING_DIR/$ENVNAME.tar.gz -C $ENVNAME
. $ENVNAME/bin/activate
tar -xzf /staging/$STAGING_DIR/code.tar.gz
tar -xzf /staging/$STAGING_DIR/$INPUT.tar.gz 
mv test videos

# run the extraction
timeout 18h python slurm.py --config-name $1
timeout_exit_status=$?
if [ $timeout_exit_status -eq 124 ]; then
    exit 85
fi
exit $timeout_exit_status

# zip the results
tar -zcvf results.tar.gz ./*.pt
rm ./*.py ./*.pt config.yaml

