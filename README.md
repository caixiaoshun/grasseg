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

|"Name"|"test/acc"|"test/dice"|"test/f1Score"|"test/iou"|"test/loss"|"test/precision"|
|---|---|---|---|---|---|---|
|"unetmobv2"|"0.5040851831436157"|"0.05965903028845787"|"0.5071227550506592"|"0.2698157727718353"|"0.7685120105743408"|"0.5218417644500732"|
|"fcn"|"0.3680895566940307"|"0.05280579254031181"|"0.3244308829307556"|"0.1847038567066193"|"0.9944534301757812"|"0.2954480051994324"|
|"farseg_resnet101"|"0.5544077754020691"|"0.06120698526501656"|"0.5661771297454834"|"0.2806955575942993"|"0.7207546234130859"|"0.6828808784484863"|
|"farseg_resnet50"|"0.5697923898696899"|"0.07711651921272278"|"0.5918323993682861"|"0.2915698289871216"|"0.6755913496017456"|"0.6980723142623901"|
|"farseg_resnet34"|"0.5361529588699341"|"0.06127778813242912"|"0.5520926713943481"|"0.25685515999794006"|"0.7473989725112915"|"0.6278958320617676"|
|"farseg_resnet18"|"0.5476192831993103"|"0.07334156334400177"|"0.5666882991790771"|"0.2743602991104126"|"0.7339965105056763"|"0.6301077604293823"|

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


