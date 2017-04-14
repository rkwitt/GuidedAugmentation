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

## RAW training data

## Pre-trained attribute models

Just execute
```bash
cd AGA
./fetch_training_data.sh
```
to download the *raw* training data for *pose* and *depth*.

## Running AGA
