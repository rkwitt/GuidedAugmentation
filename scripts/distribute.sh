#!/bin/bash

FOLDER=$1

rm -rf ${FOLDER}
mkdir ${FOLDER}

cp AGA.py 					${FOLDER}
cp utils.py 				${FOLDER}
cp AGA_full_depth.yaml		${FOLDER}
cp AGA_full_pose.yaml		${FOLDER}
cp sys_config_pose.yaml 	${FOLDER}
cp sys_config_depth.yaml	${FOLDER}
cp getmodels.sh 			${FOLDER}
cp README 					${FOLDER}

tar cvfz ${FOLDER}.tar.gz ${FOLDER}

