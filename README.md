# IDIP

A PyTorch implementation of IDIP based on the paper
[Instance-aware Diffusion Implicit Process for Instance Segmentation]().

![Network Architecture](results/arch.jpg)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

- [Detectron2](https://detectron2.readthedocs.io/en/latest/)

```
conda install tensorboard
conda install -c conda-forge pycocotools
pip install git+https://github.com/facebookresearch/detectron2.git
```

- [MMDetection](https://mmdetection.readthedocs.io/en/latest/)

```
pip install openmim
mim install mmcv 
mim install mmdet
```

## Usage

Set th environment variable DETECTRON2_DATASETS to the directory where the dataset saved, for example:
`export DETECTRON2_DATASETS=/home/data`. Then download the backbone weights
from [MEGA](https://mega.nz/folder/mSg00RZS#tkb1KdwIGZRTqcWnPZov7A), put them in `results`.

To train the model with `resnet50` backbone on COCO dataset:

```
python main.py --config-file configs/res50.yaml --num-gpus 2
```

Using tensorboard to visualize the training process:

```
tensorboard --logdir=results --bind_all
```

To evaluate the model with `resnet50` backbone on COCO dataset:

```
python main.py --config-file configs/res50.yaml MODEL.WEIGHTS results/model.pth --eval-only 
```

To visualize the results of a given image by using the pre-trained model:

```
python demo.py --config-file configs/res50.yaml --input image.jpg --output out.jpg --opts MODEL.WEIGHTS results/model.pth
```

## Benchmarks

The models are trained on two NVIDIA GeForce RTX 3090 GPUs (24G), and tested in `COCO Val 2017` dataset.
All the hyper-parameters are the default values.

| Backbone                          |  AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |                            Download                            |
|-----------------------------------|:----:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|:--------------------------------------------------------------:|
| [ResNet-50](configs/res50.yaml)   | 37.3 |      37.5       |      37.5       |      37.3      |      37.5      |      37.5      | [MEGA](https://mega.nz/folder/mSg00RZS#tkb1KdwIGZRTqcWnPZov7A) |
| [ResNet-101](configs/res101.yaml) | 41.0 |      41.1       |      41.1       |      41.0      |      41.1      |      41.1      | [MEGA](https://mega.nz/folder/mSg00RZS#tkb1KdwIGZRTqcWnPZov7A) |

## Results

![vis](results/visual.jpg)