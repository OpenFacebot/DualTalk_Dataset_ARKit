# DualTalk Dataset

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/ZiqiaoPeng/DualTalk_Dataset)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202025-blue)](https://arxiv.org/abs/2505.18096)
[![Project Page](https://img.shields.io/badge/Project%20Page-Website-green)](https://ziqiaopeng.github.io/dualtalk/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## 📖 Overview

The DualTalk dataset is the first large-scale dataset specifically designed for dual-speaker 3D talking head conversation generation. This dataset supports speaker and listener role transitions, multi-round conversations, and natural interactions, providing essential benchmark data for the 3D talking head generation field.

## 🎯 Dataset Features

- **Dual-Speaker Interaction**: Contains synchronized audio and facial expression data from two speakers
- **Role Transition**: Supports dynamic transitions between speaker and listener roles
- **Multi-Round Conversations**: Provides continuous multi-round conversation data
- **High-Quality Annotations**: Includes precise FLAME parameters and audio features
- **Diverse Scenarios**: Covers various conversation scenarios and emotional expressions

## 📁 Dataset Structure

```
DualTalk_Dataset/
├── train/
│   ├── xxx_speaker1.wav          # Speaker 1 audio files
│   ├── xxx_speaker1.npz          # Speaker 1 FLAME parameters
│   ├── xxx_speaker2.wav          # Speaker 2 audio files
│   ├── xxx_speaker2.npz          # Speaker 2 FLAME parameters
│   └── ...
├── test/
│   ├── xxx_speaker1.wav
│   ├── xxx_speaker1.npz
│   ├── xxx_speaker2.wav
│   ├── xxx_speaker2.npz
│   └── ...
└── ood/                          # Out-of-distribution test data
    ├── xxx_speaker1.wav
    ├── xxx_speaker1.npz
    ├── xxx_speaker2.wav
    ├── xxx_speaker2.npz
    └── ...
```


## 📄 Citation

If you use the DualTalk dataset, please cite our paper:

```bibtex
@inproceedings{peng2025dualtalk,
  title={Dualtalk: Dual-speaker interaction for 3d talking head conversations},
  author={Peng, Ziqiao and Fan, Yanbo and Wu, Haoyu and Wang, Xuan and Liu, Hongyan and He, Jun and Fan, Zhaoxin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21055--21064},
  year={2025}
}
```

## 📄 License

This dataset is derived from the [RealTalk dataset](https://huggingface.co/datasets/scottgeng00/realtalk) and is licensed under **Apache 2.0**. 

**Apache 2.0 License**: This license allows you to use, modify, and distribute the dataset for both commercial and non-commercial purposes, with proper attribution and license inclusion.

For the complete license text, please see: [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)

