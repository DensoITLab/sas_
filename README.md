# SAS: Structured Activation Sparsification
This is the official repo for ICLR 2024 Paper "SAS: Structured Activation Sparsification"
Yusuke Sekikawa and Shingo Yashima

[paper](https://openreview.net/pdf?id=vZfi5to2Xl), [openreview](https://openreview.net/forum?id=vZfi5to2Xl)
## Overview

## Run
### 1. Train standard ResNet-18
```bash
python train.py <TRAIN_DATA_DIR> <VAL_DATA_DIR>
```
### 2. Train ResNet-18 with SAS
```bash
python train.py <TRAIN_DATA_DIR> <VAL_DATA_DIR> --use_sas
```
### 3. Train Wide ResNet-18
```bash
python train.py <TRAIN_DATA_DIR> <VAL_DATA_DIR> --arch wide_resnet18
```
### 4. Train Wide ResNet-18 with SAS
```bash
python train.py <TRAIN_DATA_DIR> <VAL_DATA_DIR> --use_sas --arch wide_resnet18
```

## Citation
If you find our code or paper useful, please cite the following:
```
@inproceedings{
sekikawa2024sas,
title={{SAS}: Structured Activation Sparsification},
author={Yusuke Sekikawa and Shingo Yashima},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=vZfi5to2Xl}
}
```
