#!/bin/bash

# TODO: modify this part if needed
ENVNAME=feature_extraction      
INPUT_STAGING_DIR=groups/li_group_biostats
INPUT_TAR=THUMOS14_test_30fps
INPUT_FOLDER=videos_resized
OUTPUT_STAGING_DIR=whuang288
OUTPUT_TAR=clip_results_test

# Assumptions:
# 1) input directory name in the config file is set to "videos"
# 2) code is zipped to "code.tar.gz"
set -e
export PATH
mkdir $ENVNAME
tar -xzf /staging/$INPUT_STAGING_DIR/video_feature_extraction/$ENVNAME.tar.gz -C $ENVNAME
. $ENVNAME/bin/activate
tar -xzf /staging/$INPUT_STAGING_DIR/video_feature_extraction/code.tar.gz
tar -xzf /staging/$INPUT_STAGING_DIR/datasets/$INPUT_TAR.tar.gz
mv $INPUT_FOLDER videos
cp /staging/$INPUT_STAGING_DIR/datasets/video_list.txt ./videos/

# run the extraction
python main.py --config-name clip_vit_b_32

# zip the results
tar -zcvf $OUTPUT.tar.gz ./*.pt
rm ./*.py ./*.pt config.yaml
mv $OUTPUT.tar.gz /staging/$OUTPUT_STAGING_DIR/

