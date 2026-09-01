# How to download the dataset

## 1. Clone this repo (metadata & scripts)

```bash
git clone https://github.com/OpenFacebot/DualTalk_Dataset_ARKit.git
```

## 2. Download the ARKit version of DualTalk Dataset (ModelScope)

```python
# 数据集下载
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('yiwenhao/Dualtalk_Dataset_ARKit')
# 您可按需配置 subset_name、split，参照“快速使用”示例代码
```

Or via ModelScope CLI / web page: https://modelscope.cn/datasets/yiwenhao/Dualtalk_Dataset_ARKit

## 3. Download the raw DualTalk Dataset (FLAME version, optional)

Only needed if you want the original FLAME parameters or re-run the conversion.

- Hugging Face: https://huggingface.co/datasets/ZiqiaoPeng/DualTalk_Dataset
