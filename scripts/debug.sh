#!/bin/bash

/home/pma/rkwitt/torch/install/bin/th \
	../torch/pretrain.lua \
	-dataFile /scratch2/rkwitt/Mount/images/objects/desk/train.hdf5 \
	-modelFile ../torch/models/ae.lua \
	-saveModel /tmp/pretrained.t7 \
	-epochs 10 \
	-batchSize 64 \
	-cuda

/home/pma/rkwitt/torch/install/bin/th \
	../torch/train_COR.lua \
	-dataFile /scratch2/rkwitt/Mount/images/objects/desk/train.hdf5 \
	-saveCOR /tmp/deskCOR.t7 \
	-batchSize 64 \
	-epochs 20 \
	-column 1 \
	-cuda

/home/pma/rkwitt/torch/install/bin/th \
	../torch/test_COR.lua \
	-dataFile /scratch2/rkwitt/Mount/images/objects/desk/test.hdf5 \
	-modelCOR /tmp/deskCOR.t7 \
	-column 1

 /home/pma/rkwitt/torch/install/bin/th \
 	../torch/train_EDN.lua \
 	-dataFile /scratch2/rkwitt/Mount/images/objects/desk/val_i0001_cv_0001_train.hdf5 \
 	-modelEDN /tmp/pretrained.t7 \
 	-modelCOR /tmp/deskCOR.t7 \
 	-saveModel /tmp/x.t7 \
 	-target 3.5 \
 	-cuda \
 	-batchSize 64 \
 	-epochs 20

/home/pma/rkwitt/torch/install/bin/th \
	../torch/test_EDN.lua \
	-dataFile /scratch2/rkwitt/Mount/images/objects/desk/val_i0001_cv_0001_test.hdf5 \
	-model /tmp/x.t7 \
	-outputFile /tmp/y.hdf5
