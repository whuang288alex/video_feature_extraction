#!/bin/bash

# TODO: modify this part if needed
ENVNAME=feature_extraction      
INPUT_STAGING_DIR=groups/li_group_biostats
INPUT_TAR=THUMOS14_val_30fps
INPUT_FOLDER=videos_resized
OUTPUT_STAGING_DIR=whuang288
OUTPUT_TAR=clip_results_val
CONFIG_NAME=clip_vit_b_32

# Assumptions:
# input directory name in the config file is set to "videos"
set -e
export PATH
mkdir $ENVNAME
tar -xzf /staging/$INPUT_STAGING_DIR/video_feature_extraction/$ENVNAME.tar.gz -C $ENVNAME
. $ENVNAME/bin/activate
tar -xzf /staging/$INPUT_STAGING_DIR/video_feature_extraction/video_feature_extraction.tar.gz
tar -xzf /staging/$INPUT_STAGING_DIR/datasets/$INPUT_TAR.tar.gz
mv $INPUT_FOLDER videos
cp /staging/$INPUT_STAGING_DIR/datasets/video_list.txt ./videos/
python main.py --config-name $CONFIG_NAME

# zip the results
tar -zcvf $OUTPUT_TAR.tar.gz ./*.pt
rm ./*.py ./*.pt config.yaml
cp $OUTPUT_TAR.tar.gz /staging/$OUTPUT_STAGING_DIR/

