#!/bin/bash
# AA tiny script to easily rerun pencil searches to recover arguments. 
# It is intended to be run on klone and pull its arguments from environment variables.


# Usage:
# export FAKE_ID=1 # the fake ID we are performing the search on
# export FAKES_DIR=/mmfs1/home/wbeebe/pencil # the directory to store the results
# export DOWNSAMPLE_N=10 # what nthe number of images to use. If empty do not downsample.
# export CONFIG=$FAKES_DIR/$FAKE_ID/search_config_fake_$FAKE_ID.yaml # the search config file to use for the particular fake
# export WU_DIR=/mmfs1/home/wbeebe/dirac/kbmod/kbmod_wf/kbmod_new_ic/slice3/staging_42_2/results # the directory containing the WU
# export WU_FILENAME=slice3.collection.wu.42.repro # the WU file to use
# export RUN_DIR=/mmfs1/home/wbeebe/dirac/kbmod/kbmod_wf/kbmod_new_ic/kbmod-tno-scripts # the directory containing the kbmod_run_wu.py script


# Setup
echo $FAKE_ID

# Have DOWNSAMPLE_ARG be empty if DOWNSAMPLE_N is empty
if [ -z "$DOWNSAMPLE_N" ]; then
    export DOWNSAMPLE_ARG=""
else
    export DOWNSAMPLE_ARG="--downsample_n=$DOWNSAMPLE_N"
fi
 
# Prep for job
export RESULTS_DIR=$FAKES_DIR/$FAKE_ID/downsample_$DOWNSAMPLE_N
mkdir $RESULTS_DIR

echo Running: python $RUN_DIR/kbmod_run_wu.py --wu_input_file=$WU_DIR/$WU_FILENAME --sharded $DOWNSAMPLE_ARG --result_dir=$RESULTS_DIR --search_config=$CONFIG
python $RUN_DIR/kbmod_run_wu.py --wu_input_file=$WU_DIR/$WU_FILENAME --sharded $DOWNSAMPLE_ARG --result_dir=$RESULTS_DIR --search_config=$CONFIG



