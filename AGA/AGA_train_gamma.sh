#!/bin/bash

NCPU=2
BASE_DIR="/scratch2/rkwitt/data/SUNRGBD/models/"
TEST_GAMMA="../torch/test_gamma.lua"
TRAIN_GAMMA="../torch/train_gamma.lua"
MODEL_GAMMA="../torch/models/gamma.lua"


TARGET_FOLDER="depth"

# train object-agnostic gamma
th ${TRAIN_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/train.hdf5 \
    -logFile    ${BASE_DIR}/${TARGET_FOLDER}/agnosticGAMMA.log \
    -save       ${BASE_DIR}/${TARGET_FOLDER}/agnosticGAMMA.t7 \
    -model      ${MODEL_GAMMA} \
    -column     1 \
    -epochs     50 \
    -cuda
# train object-specific gamma
cat ../data/SUNRGBD/objects.txt | grep -v others | grep -v  __background__ | parallel -j ${NCPU} th \
    ${TRAIN_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/{.}/train.hdf5 \
    -logFile    ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA.log \
    -save       ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA.t7 \
    -model      ${MODEL_GAMMA} \
    -column     1 \
    -epochs     30 \
    -batchSize  64 \
    -cuda
# test object-agnostic gamma
cat ../data/SUNRGBD/objects.txt | grep -v others | grep -v  __background__ | parallel th ${TEST_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile ${BASE_DIR}/${TARGET_FOLDER}/{.}/agnosticGAMMA_predictions.hdf5 \
    -model      ${BASE_DIR}/${TARGET_FOLDER}/agnosticGAMMA.t7 \
    -column     1 \
    -eval
# test object-specific gamma
cat ../data/SUNRGBD/objects.txt | grep -v others | grep -v  __background__ | parallel th ${TEST_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA_predictions.hdf5 \
    -model      ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA.t7 \
    -column     1 \
    -eval


# TARGET_FOLDER="pose"

# train object-agnostic gamma
${TRAIN_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/train.hdf5 \
    -logFile    ${BASE_DIR}/${TARGET_FOLDER}/agnosticGAMMA.log \
    -save       ${BASE_DIR}/${TARGET_FOLDER}/agnosticGAMMA.t7 \
    -model      ${MODEL_GAMMA} \
    -column     2 \
    -epochs     50 \
    -cuda
# train object-specific gamma
cat ../data/SUNRGBD/objects.txt | grep -v others | grep -v  __background__ | parallel -j ${NCPU} th \
    ${TRAIN_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/{.}/train.hdf5 \
    -logFile    ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA.log \
    -save       ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA.t7 \
    -model      ${MODEL_GAMMA} \
    -column     2 \
    -epochs     30 \
    -batchSize  64 \
    -cuda
# test object agnostic gamma
cat ../data/SUNRGBD/objects.txt | grep -v others | grep -v  __background__ | parallel -j ${NCPU} th \
    ${TEST_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile ${BASE_DIR}/${TARGET_FOLDER}/{.}/agnosticGAMMA_predictions.hdf5 \
    -model      ${BASE_DIR}/${TARGET_FOLDER}/agnosticGAMMA.t7 \
    -column     2 \
    -eval \
# test object-specific gamma
cat ../data/SUNRGBD/objects.txt | grep -v others | grep -v  __background__ | parallel -j ${NCPU} th \
    ${TEST_GAMMA} \
    -dataFile   ${BASE_DIR}/${TARGET_FOLDER}/{.}/test.hdf5 \
    -outputFile ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA_predictions.hdf5 \
    -model      ${BASE_DIR}/${TARGET_FOLDER}/{.}/objectGAMMA.t7 \
    -column     2 \
    -eval \