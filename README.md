# Unpacking the Gap Box Against Data-Free Knowledge Distillation
This repository is the official code for the paper "Unpacking the Gap Box Against Data-Free Knowledge Distillation" by Yang Wang, Biao Qian, Haipeng Liu, Yong Rui and Meng Wang.

### Dependencies

Python 3.6
PyTorch 1.2.0
dependencies in requirements.txt


### Installation
Install pytorch and other dependencies:

        pip install -r requirements.txt


### Set the paths of datasets

Set the "data_root" in "datafree_kd.py" as the path root of your dataset. For example:

        data_root = "/home/Datasets/"


### Training

To train MobileNetV2 (student model) with ResNet-34 (teacher model) on CIFAR-100, run the following command:

    bash scripts/gapssg/cifar100_resnet34_mobilenetv2.sh



