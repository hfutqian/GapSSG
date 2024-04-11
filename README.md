# Unpacking the Gap Box Against Data-Free Knowledge Distillation [T-PAMI 2024]
This repository is the official code for the paper "Unpacking the Gap Box Against Data-Free Knowledge Distillation" by Yang Wang, Biao Qian, Haipeng Liu, Yong Rui and Meng Wang.

## Introduction
We study the **gap disturbance issue** between teacher (T) and student (S) in knowledge distillation (KD), which has attracted increasing attention under data-driven setting since it enables the understanding on why KD works well and improves their effectiveness. Unlike data-driven setting, the significant **challenges** under data-free scenario lie in: when the generated samples are "good" for S with the gap disturbance; how to yield the ideal generated samples by defeating the gap issue, to maximally benefit S. To sum up, our major contributions are summarized as follows:



(1) We deliver **the first attempt** to study how the gap disturbance between T and S affects KD process in data-free scenario. Following that, the **theoretical analysis** suggests the existence of an **ideal teacher** T* and an upper bound for the generalization gap of S, implying that generalization heavily relies on the mismatch between T and the ideal teacher T*, which serves as the theoretical basis for generating "good" samples.

(2) Motivated by the above, we further propose to unpack the **reality gap box** between T and S into the **derived** and **inherent** gap, which helps generate the "**good**" samples to maximally benefit S with the fixed T by defeating the gap disturbance.

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
If you find the project codes useful for your research, please consider citing
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


