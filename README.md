<div align="center">

# Grass segmentation for grass coverage


</div>

<br>

# Introduction

This repository contains the code for the Grass segmentation for grass coverage estimation. The code is written in Python and uses the PyTorch framework.

# Installation

To install the required packages, run the following command:

```
pip install -r requirements.txt

pip install -e .
```

# Datastet

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

# method

[farseg](references/Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_Resolution_Remote_Sensing_Imagery.pdf)

# experiment

|       **Name**       |   **acc**  |  **dice**  | **f1Score** |   **iou**  | **precision** |
|:--------------------:|:----------:|:----------:|:-----------:|:----------:|:-------------:|
|        **fcn**       |   0.3681   |   0.0528   |    0.3244   |   0.1847   |     0.2954    |
|     **unetmobv2**    |   0.5041   |   0.0596   |    0.5071   |   0.2698   |     0.5218    |
|  **farseg_resnet34** |   0.5362   |   0.0613   |    0.5521   |   0.2569   |     0.6279    |
|  **farseg_resnet18** |   0.5476   |   0.0733   |    0.5667   |   0.2744   |     0.6301    |
| **farseg_resnet101** |   0.5544   |   0.0612   |    0.5662   |   0.2807   |     0.6829    |
|  **farseg_resnet50** | **0.5698** | **0.0771** |  **0.5918** | **0.2916** |   **0.6981**  |

![model_eval](https://github.com/user-attachments/assets/687ce2f7-e348-4b15-bb4c-850d31992276)


# Usage

To train the model, run the following command:

```
python src/train.py experiment=fcn
```

To test the model, run the following command:

```
python src/test.py
```


# References

- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)


