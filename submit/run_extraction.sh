#!/bin/bash

# TODO: modify this part if needed
ENVNAME=feature_extraction      
INPUT_STAGING_DIR=groups/li_group_biostats
INPUT_TAR=THUMOS14_test
INPUT_FOLDER=test
OUTPUT_STAGING_DIR=groups/li_group_biostats
OUTPUT_TAR=results

# Assumptions:
# 1) input directory name in the config file is set to "videos"
# 2) code is zipped to "code.tar.gz"
set -e
export PATH
mkdir $ENVNAME
tar -xzf /staging/$INPUT_STAGING_DIR/$ENVNAME.tar.gz -C $ENVNAME
. $ENVNAME/bin/activate
tar -xzf /staging/$INPUT_STAGING_DIR/code.tar.gz
tar -xzf /staging/$INPUT_STAGING_DIR/$INPUT_TAR.tar.gz
mv $INPUT_FOLDER videos

# run the extraction
timeout 23h python slurm.py --config-name $1
timeout_exit_status=$?
if [ $timeout_exit_status -eq 124 ]; then
    exit 85
fi
exit $timeout_exit_status

# zip the results
tar -zcvf $OUTPUT.tar.gz ./*.pt
rm ./*.py ./*.pt config.yaml
mv $OUTPUT.tar.gz /staging/$OUTPUT_STAGING_DIR/

