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
