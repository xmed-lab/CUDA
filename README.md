# CUDA: Concept-Based Unsupervised Domain Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Concept-Based Unsupervised Domain Adaptation"** (ICML 2025).

## Overview

CUDA (Concept-Based Unsupervised Domain Adaptation) is a novel approach for unsupervised domain adaptation that leverages concept-based learning to improve domain transfer performance. The method combines domain adversarial training with concept-aware feature learning to achieve better alignment between source and target domains.

## Project Structure

```
CUDA/
├── main.py                    # Main training and evaluation script
├── utils.py                   # Utility functions and data loading
├── models/
│   └── domain_adversarial_network.py  # Domain discriminator implementation
├── alignemnt/
│   └── dann.py               # Domain adversarial loss and classifier
├── data/                     # Dataset implementations
│   ├── MNIST/
│   ├── Skincon/
│   └── Waterbirds/
├── analysis/                 # Analysis and visualization tools
│   ├── __init__.py          # Feature collection utilities
│   ├── a_distance.py        # A-distance calculation
│   ├── pca.py              # PCA analysis
│   └── tsne.py             # t-SNE visualization
└── README.md
```

## Installation

```bash
# Install PyTorch with CUDA support
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install timm==0.6.13
pip install numpy==1.26.4
pip install matplotlib==3.5.2
pip install --extra-index-url https://pypi.org/simple/ -i https://test.pypi.org/simple/ tllib==0.4
pip install scikit-learn
pip install wandb  # for experiment tracking
```

## Usage

### Basic Training

```bash
python main.py --data waterbirds \
               --root ./data/Waterbirds/waterbirds-dann-2 \
               --epochs 50 \
               --batch-size 32 \
               --lr 1e-3 \
               --lambda_c 5.0 \
               --lambda_t 0.3 \
               --tau 0.7
```

### Key Parameters

- `--data`: Dataset choice (`waterbirds`, `mnist`, `skincon`)
- `--lambda_c`: Trade-off parameter for concept loss 
- `--lambda_t`: Trade-off parameter for transfer loss 
- `--tau`: Relax threshold parameter for transfer loss 
- `--bottleneck_dim`: Bottleneck dimension 
- `--concept_emb_dim`: Concept embedding dimension 

### Evaluation and Analysis

```bash
# Test mode
python main.py --phase test --root ./data --data waterbirds

# Analysis mode with visualizations
python main.py --phase analysis --root ./data --data waterbirds

# Concept analysis
python main.py --phase concept-analysis --root ./data --data waterbirds
```

## Supported Datasets

1. **Waterbirds, CUB**: Bird classification with spurious background correlations
2. **MNIST, MNIST-M, SVHN, USPS**: Handwritten digit recognition with domain shifts
3. **Skincon**: Skin condition classification across different skin tones

## Citation

If you use this code in your research, please cite:

```bibtex
@article{CUDA,
  title={Concept-Based Unsupervised Domain Adaptation},
  author={Xu, Xinyue and Hu, Yueying and Tang, Hui and Qin, Yi and Mi, Lu and Wang, Hao and Li, Xiaomeng}
}
```

## Acknowledgments

- Built upon the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)
- Incorporates concepts from Domain Adversarial Neural Networks (DANN)
- CUB dataset implementation adapted from [ConceptBottleneck](https://github.com/yewsiang/ConceptBottleneck)
- Waterbrid dataset split adapted from [CONDA](https://github.com/jihyechoi77/CONDA)

