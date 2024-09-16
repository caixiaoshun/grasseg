# Grass segmentation for grass coverage estimation
[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/caixiaoshun/cloudseg)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/XavierJiezou/cloudseg#license)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

## Introduction

This repository contains the code for the Grass segmentation for grass coverage estimation. The code is written in Python and uses the PyTorch framework.

## Installation

To install the required packages, run the following command:

```
pip install -r requirements.txt

pip install -e .
```

## Datastet

```bash
cloudseg
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

## method

- [fcn (CVPR 2015)](references/Fully-Convolutional-Networks-for-Semantic-Segmentation.pdf)
- [unet (MICCAI 2015)](references/U-Net-Convolutional-Networks-for-Biomedical-Image-Segmentation.pdf)
- [pspnet (CVPR(2017)](references/Pyramid-Scene-Parsing-Network.pdf)
- [unet plus plus (MICCAI 2015)](references/U-Net-Convolutional-Networks-for-Biomedical-Image-Segmentation.pdf)
- [farseg (CVPR 2020)](references/Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_Resolution_Remote_Sensing_Imagery.pdf)


## experiment

|       **Name**       |   **acc**  |  **dice**  | **f1Score** |   **iou**  | **precision** |
|:--------------------:|:----------:|:----------:|:-----------:|:----------:|:-------------:|
|        **fcn**       |   0.3681   |   0.0528   |    0.3244   |   0.1847   |     0.2954    |
|  **farseg_resnet34** |   0.5362   |   0.0613   |    0.5521   |   0.2569   |     0.6279    |
|  **farseg_resnet18** |   0.5476   |   0.0733   |    0.5667   |   0.2744   |     0.6301    |
| **farseg_resnet101** |   0.5544   |   0.0612   |    0.5662   |   0.2807   |     0.6829    |
|  **farseg_resnet50** | **0.5698** | **0.0771** |  **0.5918** | **0.2916** |   **0.6981**  |

![model_eval](https://github.com/user-attachments/assets/687ce2f7-e348-4b15-bb4c-850d31992276)


## Usage

To train the model, run the following command:

```
python src/train.py experiment=fcn
```

To test the model, run the following command:

```
python src/test.py
```


## References

- [PyTorch](https://pytorch.org/)
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)


