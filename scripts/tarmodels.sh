#!/bin/bash

FOLDER=$1
TARGET=$3
MODELS=$2

CWD=`pwd`
cd ${FOLDER}
find ${MODELS} -name '*.t7' -print0 | tar -cvf ${TARGET} -T -
cd ${CWD}
