# Unpacking the Gap Box Against Data-Free Knowledge Distillation [T-PAMI 2024]
This repository is the official code for the paper "Unpacking the Gap Box Against Data-Free Knowledge Distillation" by Yang Wang, Biao Qian, Haipeng Liu, Yong Rui and Meng Wang.

## Dependencies

* Python 3.6
* PyTorch 1.2.0
* Dependencies in requirements.txt

## Usages

### Installation
Install pytorch and other dependencies:

        pip install -r requirements.txt


### Set the paths of datasets

Set the "data_root" in "datafree_kd.py" as the path root of your dataset. For example:

        data_root = "/home/Datasets/"


### Training

To train MobileNetV2 (student model) with ResNet-34 (teacher model) on CIFAR-100, run the following command:

    bash scripts/gapssg/cifar100_resnet34_mobilenetv2.sh


## Citation
If you find the codes useful for your research, please consider citing
```
@ARTICLE{10476709,
  author={Wang, Yang and Qian, Biao and Liu, Haipeng and Rui, Yong and Wang, Meng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Unpacking the Gap Box Against Data-Free Knowledge Distillation}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Training;Art;Data models;Analytical models;Knowledge engineering;Generators;Three-dimensional displays;Data-free knowledge distillation;derived gap;empirical distilled risk;generative model;inherent gap},
  doi={10.1109/TPAMI.2024.3379505}}

```


