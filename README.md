# AGA : Attribute-Guided Augmentation

This repository contains *code* and *data* for the following manuscript
(please use this citation format when using the code):

*Disclaimer*: Code for using pretrained models is already online. Training
code will be released ASAP.

```
@inproceedings{Dixit17a,
    author    = {M.~Dixit and R.~Kwitt and M.~Niethammer and N.~Vasconcelos},
    title     = {AGA : Attribute-Guided Augmentation},
    booktitle = {CVPR},
    year      = 2017}
```

## Introduction

AGA is an augmentation technique in feature space that *learns how
features change as a function of some auxiliary attribute*.

## Using pre-trained attribute models

We will use `/scratch` as our base directory. 
To use **pre-trained** (on SUN RGB-D) *pose* and *depth* models for AGA, first
download the models using:

```bash
cd AGA
./fetch_pretrained_models.sh
mkdir -p /scratch/models
mv depth_models_04142017.tar /scratch/models
mv pose_models_04142017.tar /scratch/models
cd /scratch/models
tar xvf depth_models_04142017.tar
tar xvf pose_models_04142017.tar
```
This will create directories `/scratch/pose` and `/scratch/depth` which contain
all required files.

## Checking out the code

Use
```bash
cd /scratch
git clone https://github.com/rkwitt/GuidedAugmentation.git
```
to check out the code from the Git repository.

## Example: AGA

To show an example of how AGA can be used to synthesize features, we use the
pre-trained models from above and download a set of images with labels, bounding
boxes and Fast-RCNN FC7 features.

```bash
cd /scratch
wget www.cosy.sbg.ac.at:/~rkwitt/AGA/T1.tar.gz
tar xvfz T1.tar.gz
```

We then copy the file `sys_config_depth_template.yaml` to
`sys_config_depth.yaml`, i.e.,
```bash
cd /scratch/GuidedAugmentation/config
cp sys_config_depth_template.yaml sys_config_depth.yaml
```
and edit `sys_config_depth.yaml` according to our system setup:
```bash
{
TRAIN_PRE:      /scratch/GuidedAugmentation/torch/pretrain.lua,
TEST_GAMMA:     /scratch/GuidedAugmentation/torch/test_gamma.lua,
TRAIN_PHI:      /scratch/GuidedAugmentation/torch/train_phi.lua,
TEST_PHI:       /scratch/GuidedAugmentation/torch/test_phi.lua,
PHI_DEF:        /scratch/GuidedAugmentation/torch/models/phi.lua,
GAMMA_DEF:      /scratch/GuidedAugmentation/torch/models/gamma.lua,
TORCH:          /opt/torch/install/bin/th,
PATH_TO_MODELS: /scratch/models,
TEMP_DIR:       /tmp/
}
```
We are now ready to run `AGA.py` on our image features in the `T1` folder. For this, we
create a file which lists the full path of all images as
```bash
find /scratch/T1 \
    -name '*.jpg' -exec sh -c 'printf "%s\n" "${0%.*}"' {} ';' > T1_list.txt
```
Alternatively, you can use
```bash
find /scratch/T1 -name '*.jpg' | python strip_extension.py > T1_list.txt
```
Finally, run **AGA** (using all threads in your system) to synthesize
features:
```bash
parallel python AGA.py \
    --sys_config ../config/sys_config_depth.yaml \
    --aga_config /scratch/models/depth/AGA_output.yaml \
    --input_file {.}_bbox_features.mat \
    --label_file {.}_ss_labels.mat \
    --output_file {.}_bbox_features_AGA_depth.mat \
    --verbose ::: `cat /scratch/T1_list.txt`
```
