# GrassEG - Grassland Semantic Segmentation
[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/caixiaoshun/cloudseg)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/XavierJiezou/cloudseg#license)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

## Introduction

**GrassEG** is a PyTorch-based deep learning model for semantic segmentation of grassland images. It is designed to classify grass coverage into five categories (low, medium-low, medium, medium-high, high) and can handle high-resolution satellite or aerial imagery.

## Table of Contents

- [Introduction](#Installation)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Methods](#methods)
- [Results](#results)
- [References](#references)
- [License](#license)

## Introduction

GrassEG aims to provide a reliable deep learning framework for the segmentation of grassland imagery. This project addresses a key problem in environmental monitoring and grassland management, allowing the classification of vegetation coverage in satellite or drone images.

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/caixiaoshun/grasseg.git
cd grasseg
pip install -r requirements.txt
pip install -e .
```
Ensure you have PyTorch, torchvision, and other required libraries installed.

## Usage

### Dataset Preparation

Organize your dataset with the following structure:

```bash
grasseg
├── src
├── configs
├── ...
├── data
│   ├── grass
│   │   ├── train
│   │   │   ├── img
│   │   │   ├── ann
│   │   ├── val
│   │   │   ├── img
│   │   │   ├── ann
```

### Training

To train the model, run the following command:

```bash
python src/train.py experiment=farseg-resnet50
```

This script supports different configuration files for models like Swin Transformer or FCN. You can adjust hyperparameters such as learning rate, batch size, and epochs in the configuration file.

### Evaluation

Evaluate the trained model using:

```bash
python src/eval/vis_model.py
```

The evaluation will output segmentation metrics including Dice coefficient and Intersection over Union (IoU).

## Method

- [fcn (CVPR 2015)](references/Fully-Convolutional-Networks-for-Semantic-Segmentation.pdf)
- [unet (MICCAI 2015)](references/U-Net-Convolutional-Networks-for-Biomedical-Image-Segmentation.pdf)
- [pspnet (CVPR(2017)](references/Pyramid-Scene-Parsing-Network.pdf)
- [fpn (CVPR(2017)](references/A-Unified-Architecture-for-Instance-and-Semantic-Segmentation.pdf)
- [linknet (VCIP 2017)](references/LinkNet-Exploiting-Encoder-Representations-for-Efficient-Semantic-Segmentation.pdf)
- [deeplabv3 (arxiv 2017)](references/Rethinking-Atrous-Convolution-for-Semantic-Image-Segmentation.pdf)
- [deeplabv3+ (ECCV 2018)](references/Encoder-Decoder-with-Atrous-Separable-Convolution-for-Semantic-Image-Segmentation.pdf)
- [unet plus plus (MICCAI 2018)](references/UNet++-A-Nested-U-Net-Architecture-for-Medical-Image-Segmentation.pdf)
- [pan (arxiv 2018)](references/Pyramid_attention_network_for_semantic_segmentation.pdf)
- [farseg (CVPR 2020)](references/Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_Resolution_Remote_Sensing_Imagery.pdf)
- [manet (IEEE Access (2020))](references/MA-Net_A_Multi-Scale_Attention_Network_for_Liver_and_Tumor_Segmentation.pdf)

## Results

| **method**        | **backbone**         | **mAcc**   | **mDice**  | **mF1Score** | **mIoU**   | **mPrecision** | **mCrossEntropyLoss** |
|-------------------|----------------------|------------|------------|--------------|------------|----------------|-----------------------|
| **deeplabv3**     | resnet152            | 0.5521     | 0.0633     | 0.5677       | 0.2715     | 0.6041         | 0.6246                |
| **pspnet**        | timm-efficientnet-l2 | 0.5497     | 0.0664     | 0.5645       | 0.2705     | 0.6154         | 0.6063                |
| **fcn**           | resnet50            | 0.5555     | 0.0680     | 0.5738       | 0.2731     | 0.6072         | 0.6182                |
| **unet++**        | se_resnext101_32x4d  | 0.5628     | 0.0636     | 0.5858       | 0.2826     | 0.6452         | 0.5878                |
| **pan**           | se_resnext101_32x4d  | 0.5681     | 0.0839     | 0.5905       | 0.2828     | 0.6305         | 0.6029                |
| **unet**          | timm-efficientnet-l2 | 0.5560     | 0.0641     | 0.5797       | 0.2808     | **0.6467**     | 0.5935                |
| **farseg**        | resnet50             | 0.5680     | 0.0801     | 0.5914       | 0.2794     | 0.6422         | 0.6163                |
| **deeplabv3plus** | timm-efficientnet-l2 | 0.5526     | **0.0845** | 0.5752       | 0.2795     | 0.6397         | 0.5823                |
| **manet**         | se_resnext101_32x4d  | 0.5642     | 0.0732     | 0.5867       | 0.2854     | 0.6375         | 0.6141                |
| **fpn**           | timm-regnetx_320     | 0.5663     | 0.0743     | 0.5849       | 0.2847     | 0.6267         | 0.5937                |
| **linknet**       | timm-resnest101e     | **0.5734** | 0.0747     | **0.5924**   | **0.2919** | 0.6330         | **0.5849**            |

![model_eval](https://github.com/user-attachments/assets/687ce2f7-e348-4b15-bb4c-850d31992276)


## References

- [PyTorch](https://pytorch.org/)
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## License

This project is licensed under the MIT License.

