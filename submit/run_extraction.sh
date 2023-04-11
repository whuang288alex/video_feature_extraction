#!/bin/bash

# TODO: modify this part if needed
ENVNAME=feature_extraction      
INPUT_STAGING_DIR=groups/li_group_biostats
OUTPUT_STAGING_DIR=whuang288
INPUT=THUMOS14_test
OUTPUT=filtered_test

# Assume input directory name is set to "videos" and code is zipped to "code.tar.gz"
set -e
export PATH
mkdir $ENVNAME
tar -xzf /staging/$INPUT_STAGING_DIR/$ENVNAME.tar.gz -C $ENVNAME
. $ENVNAME/bin/activate
tar -xzf /staging/$INPUT_STAGING_DIR/code.tar.gz
tar -xzf /staging/$INPUT_STAGING_DIR/$INPUT.tar.gz
mv test videos

# run the extraction
timeout 18h python slurm.py --config-name $1
timeout_exit_status=$?
if [ $timeout_exit_status -eq 124 ]; then
    exit 85
fi
exit $timeout_exit_status

# zip the results
tar -zcvf $OUTPUT.tar.gz ./*.pt
rm ./*.py ./*.pt config.yaml
mv $OUTPUT.tar.gz /staging/$OUTPUT_STAGING_DIR/

