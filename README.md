# AGA : Attribute-Guided Augmentation

This repository contains *code* and *data* for the following manuscript
(please use this citation when using the code):

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

## Requirements

AGA is implemented in Torch and Python and uses a modified version of
[Fast-RCNN](https://github.com/rbgirshick/fast-rcnn).

## Data

First, download the SUN RGB-D dataset [here](http://rgbd.cs.princeton.edu/).


## Preprocessing

### Setup

In the following, we assume a directory setup:
```
--/scratch/
     |--SUNRGBD/
           |--SUNRGBDMeta_correct2D.mat
           |--images/
```

Download the `SUNRGBDMeta_correct2D.mat` file from the SUN RGB-D website and
create a configuration structure in MATLAB as follows:
```matlab
config.SUNRGBD_dir = '/scratch/SUNRGBD'
config.outdir      = '/scratch/SUNRGBD/images'
```
Then, run
```matlab
SUNRGBD_bbox_label
```
This will (1) copy all image files to `config.outdir`; (2) extract *selective
search* bounding boxes and (3) extract pose + depth for every bounding box
that overlaps with a ground truth bounding box by IoU>=0.5.
All this data will be saved to a cell array `MetaData` (which is also saved
to disk as `<config.outdir>/MetaData.mat`).


### Running Fast-RCNN
Next, we can run the (fine-tuned) Fast-RCNN detector on all the SUN RGB-D images
and extract FC7 features.


### Generating training data

## Training

## Running AGA
