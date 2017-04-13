#!/bin/bash

#
# Config
#
NCPU=2

########## POSE ##########

TARGET_FOLDER="reproduce_pose"

#
# Train object-agnostic POSE ARs
#
th ../torch/train_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/train.hdf5 \
    -logFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/agnosticAR.log \
    -save /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/agnosticAR.t7 \
    -regModel ../torch/models/ar.lua \
    -column 2 \
    -cuda \
    -epochs 50
#
# Train object-specific POSE ARs
#
cat SUNRGBD_objects.txt | grep -v others | grep -v  __background__ | parallel -j ${NCPU} th \
    ../torch/train_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/train.hdf5 \
    -logFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR.log \
    -save /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR.t7 \
    -regModel ../torch/models/ar.lua \
    -column 2 \
    -epochs 30 \
    -batchSize 64 \
    -cuda
#
# Test object-agnostic POSE ARs
#
cat SUNRGBD_objects.txt | grep -v others | grep -v  __background__ | parallel th ../torch/test_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/agnosticAR_predictions.hdf5 \
    -model /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/agnosticAR.t7 \
    -column 2 \
    -eval \


cat SUNRGBD_objects.txt | grep -v others | grep -v  __background__ | parallel th ../torch/test_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR_predictions.hdf5 \
    -model /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR.t7 \
    -column 2 \
    -eval \


########## DEPTH ##########

TARGET_FOLDER="reproduce_depth"

#
# Train object-agnostic DEPTH ARs
#
th ../torch/train_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/train.hdf5 \
    -logFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/agnosticAR.log \
    -save /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/agnosticAR.t7 \
    -regModel ../torch/models/ar.lua \
    -column 1 \
    -cuda \
    -epochs 50
#
# Train object-specific DEPTH ARs
#
cat SUNRGBD_objects.txt | grep -v others | grep -v  __background__ | parallel -j ${NCPU} th \
    ../torch/train_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/train.hdf5 \
    -logFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR.log \
    -save /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR.t7 \
    -regModel ../torch/models/ar.lua \
    -column 1 \
    -epochs 30 \
    -batchSize 64 \
    -cuda
#
# Test object-agnostic DEPTH ARs
#
cat SUNRGBD_objects.txt | grep -v others | grep -v  __background__ | parallel th ../torch/test_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/agnosticAR_predictions.hdf5 \
    -model /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/agnosticAR.t7 \
    -column 1 \
    -eval

#
# Test object-specific DEPTH ARs
#
cat SUNRGBD_objects.txt | grep -v others | grep -v  __background__ | parallel th ../torch/test_AR.lua \
    -dataFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR_predictions.hdf5 \
    -model /scratch2/rkwitt/data/SUNRGBD/models/${TARGET_FOLDER}/{.}/objectAR.t7 \
    -column 1 \
    -eval
