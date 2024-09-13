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

# experiment

![model eval](images/model_eval.png)

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


