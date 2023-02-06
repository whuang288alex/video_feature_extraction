#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e
ENVNAME=feature_extraction
ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

tar -xzf code.tar.gz

# modify this line to run your desired Python script and any other work you need to do
timeout 4h python slurm.py --config-name $1