# CKA-framework
This repository is an official implementation of the paper "Joint lesion detection and classification of breast 
ultrasound video via a clinical knowledge-aware framework". (The complete code will be added after the manuscript is published.)

## Dataset
We collect and annotate an ultrasound video
dataset (298 videos) for lesion detection and classification in ultrasound videos, including 170 benign cases and 128 
malignant cases. This dataset is available for only non-commercial use in research or educational
purpose. As long as you use the dataset for these purposes, you can edit or process images and annotations in the dataset.
(Note that full dataset will be made publicly available upon the publication of the manuscript)

## Code Usage

## Installation

### Requirements

* Linux, CUDA>=11.3, GCC>=7.5.0
  
* Python>=3.8

* PyTorch>=1.11.0, torchvision>=0.12.0 (following instructions [here](https://pytorch.org/))

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
  
### Dataset preparation

Please organize the dataset as following:

```
code_root/
└── buv/
      ├── rawframes/
      ├── train.json
      └── val.json
```

### Training

For example, the command for training CKA-framework is as following:

```bash
python main.py
```
The configs in main.py can be changed.

### Evaluation

After obtaining the trained CKA-framework, then run following command to evaluate it on the validation set:

```bash
python validation.py
```

## Notes
The code of this repository is built on
https://github.com/fundamentalvision/Deformable-DETR.
