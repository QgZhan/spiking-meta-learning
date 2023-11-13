# spiking meta-learning

This repository contains the code for ''A Two-stage Spiking Meta-learning Method for Few-shot Classification''.


## Main Results

*The models on *Omniglot* use SNN-ConvNet-4 as backbone.*

#### accuracy (%) on *Omniglot*

method| 2-way 1-shot|2-way 5-shot| 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
:-:|:----------:|:-:|:------------:|:------------:|:-------------:|:-------------:
CESM |96.71±0.32|98.59±0.28|90.90 ±0.38|96.89±0.55|78.99±0.68|90.63±0.18|
MESM | 96.75±0.32|98.63±0.26|94.86±0.32|97.99±0.36|84.87±0.48|93.42±0.16|

*The models on *miniImageNet* and *tieredImageNet* use SNN-ResNet-12 as backbone, the channels in each block are **64-128-256-512**, the backbone does **NOT** introduce any additional trick (e.g. DropBlock or wider channel in some recent work).*

#### accuracy (%) on *miniImageNet*

method| 2-way 1-shot|2-way 5-shot| 5-way 1-shot | 5-way 5-shot | 10-way 1-shot | 10-way 5-shot | 20-way 1-shot | 20-way 5-shot |
:-:|:----------:|:-:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
CESM |75.13±0.30|84.65±0.21|48.37±0.24|65.61±0.26|34.26±0.26|52.60±0.65|23.14±0.42|38.94±0.43|
MESM |74.56±0.24|84.68±0.19|51.54±0.23|69.94±0.18|35.87±0.84|53.83±0.79|25.03±0.47|41.08±0.44|

#### accuracy (%) on *tieredImageNet*

method| 2-way 1-shot|2-way 5-shot| 5-way 1-shot | 5-way 5-shot |
:-:|:----------:|:-:|:------------:|:------------:|
CESM |75.98±0.21|86.46±0.26|52.51±0.22|68.66±0.28|
MESM |76.59±0.18|88.49±0.13|53.76±0.15|69.01±0.16|
####

## Running the code

### Preliminaries

**Environment**
- Python 3.7.3
- Pytorch 1.2.0
- tensorboardX

**Datasets**
- [Omniglot]
- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))

### 1. Training Classifier-Baseline
```
python train_cesm.py --config configs/train_classifier_mini.yaml
```

### 2. Training Meta-Baseline
```
python train_mesm.py --config configs/train_meta_mini.yaml
```

### 3. Test
To test the performance, modify `configs/test_few_shot.yaml` by setting `load_encoder` to the saving file of Classifier-Baseline, or setting `load` to the saving file of Meta-Baseline.

E.g., `load: ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/max-va.pth`

Then run
```
python test_few_shot.py --shot 1
```
