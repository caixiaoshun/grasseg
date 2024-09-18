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
- [farseg (CVPR 2020)](references/Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_Resolution_Remote_Sensing_Imagery.pdf)
- [manet (IEEE Access (2020))](references/MA-Net_A_Multi-Scale_Attention_Network_for_Liver_and_Tumor_Segmentation.pdf)

## Results

|       **Name**       |   **acc**  |  **dice**  | **f1Score** |   **iou**  | **precision** |
|:--------------------:|:----------:|:----------:|:-----------:|:----------:|:-------------:|
|  **farseg_resnet34** |   0.5362   |   0.0613   |    0.5521   |   0.2569   |     0.6279    |
|  **farseg_resnet18** |   0.5476   |   0.0733   |    0.5667   |   0.2744   |     0.6301    |
| **farseg_resnet101** |   0.5544   |   0.0612   |    0.5662   |   0.2807   |     0.6829    |
|  **farseg_resnet50** | **0.5698** | **0.0771** |  **0.5918** | **0.2916** |   **0.6981**  |

![model_eval](https://github.com/user-attachments/assets/687ce2f7-e348-4b15-bb4c-850d31992276)


## References

- [PyTorch](https://pytorch.org/)
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## License

This project is licensed under the MIT License.

