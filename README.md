# DualTalk Dataset — ARKit Version

[![ModelScope Datasets](https://img.shields.io/badge/ModelScope-Datasets-4E82EE)](https://modelscope.cn/datasets/yiwenhao/Dualtalk_Dataset_ARKit)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/ZiqiaoPeng/DualTalk_Dataset)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202025-blue)](https://arxiv.org/abs/2505.18096)
[![Project Page](https://img.shields.io/badge/Project%20Page-Website-green)](https://ziqiaopeng.github.io/dualtalk/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repo converts the [DualTalk](https://huggingface.co/datasets/ZiqiaoPeng/DualTalk_Dataset) dataset (FLAME parameters) into the **ARKit 61 blendshape** version, so that it can be directly used for ARKit-rig based rendering and training.

> ⚠️ The conversion is a **linear mapping** from FLAME to ARKit, so some error is unavoidable. Visualization shows the converted results are acceptable.

## 📖 Overview

DualTalk is the first large-scale dataset specifically designed for dual-speaker 3D talking head conversation generation. It supports speaker/listener role transitions, multi-round conversations, and natural interactions. This ARKit version additionally provides per-frame **61-dimensional ARKit blendshape weights** for every sample, making it plug-and-play for ARKit-based pipelines.

## 🎯 Dataset Features

- **Dual-Speaker Interaction**: synchronized audio and facial expression data from two speakers
- **Role Transition**: dynamic transitions between speaker and listener roles
- **Multi-Round Conversations**: continuous multi-round conversation data
- **ARKit Blendshapes**: 61 ARKit blendshape weights per frame (converted from FLAME)
- **High-Quality Annotations**: FLAME parameters, ARKit blendshapes, audio and transcripts, with per-sample metadata and checksums

## ⬇️ Download

See [DOWNLOAD.md](./DOWNLOAD.md).

Quick start (ModelScope):

```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('yiwenhao/Dualtalk_Dataset_ARKit')
# 您可按需配置 subset_name、split，参照“快速使用”示例代码
```

## 🚀 Reproduce the Conversion

```bash
git clone https://github.com/OpenFacebot/DualTalk_Dataset_ARKit.git
cd DualTalk_Dataset_ARKit/script
python gen_metadata.py           # generate metadata/*.jsonl
bash convert_flame2arkit.sh      # convert FLAME -> ARKit blendshapes
```

Converted ARKit data is saved under `ARKit_npy/{train,test,ood}/`, and per-sample metadata (text / audio / flame / arkit paths) is written to `metadata/*.jsonl`.

## 📁 Dataset Structure

```
DualTalk_Dataset_ARKit/
├── train/                          # raw DualTalk data (FLAME version)
│   ├── xxx_speaker1.wav            # speaker 1 audio
│   ├── xxx_speaker1.npz            # speaker 1 FLAME parameters
│   ├── xxx_speaker1.txt            # speaker 1 transcript
│   ├── xxx_speaker2.wav / .npz / .txt
│   └── ...
├── test/                           # same layout as train/
├── ood/                            # out-of-distribution test data
├── ARKit_npy/                      # converted ARKit blendshapes
│   ├── train/xxx_speaker1.npy      # (T, 61) ARKit blendshape weights
│   ├── test/...
│   └── ood/...
├── metadata/                       # per-split sample index
│   ├── train.jsonl
│   ├── test.jsonl
│   └── ood.jsonl
└── script/                         # conversion scripts
    ├── gen_metadata.py
    ├── convert_flame2arkit.py / .sh
    └── convert_arkit_for_render.py
```

Each line of `metadata/*.jsonl` describes one sample:

```json
{
  "name": "xxx_sub_video_25_000",
  "speaker1_text": "...",
  "speaker1_audio": "train/xxx_speaker1.wav",
  "speaker1_flame": "train/xxx_speaker1.npz",
  "speaker1_arkit": "train/xxx_speaker1.npy",
  "speaker2_text": "...",
  "speaker2_audio": "train/xxx_speaker2.wav",
  "speaker2_flame": "train/xxx_speaker2.npz",
  "speaker2_arkit": "train/xxx_speaker2.npy",
  "speaker2_frames": 750,
  "...": "sha256 checksums, smoothing / head-correction flags, etc."
}
```

Loading an ARKit blendshape sequence:

```python
import numpy as np
arkit = np.load("ARKit_npy/train/xxx_speaker1.npy")   # (T, 61)
```

## 📄 Citation

If you use the DualTalk dataset, please cite the original paper:

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
