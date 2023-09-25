
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



