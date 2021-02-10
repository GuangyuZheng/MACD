# MACD
### Implementation for EMNLP 2020 "Unsupervised Natural Language Inference via Decoupled Multimodal Contrastive Learning"

# Requirements
- Python (>=3.5)
- torch (>=1.1.0)
- [transformers (>=2.3.0)](https://github.com/huggingface/transformers)

# Get Required Data
- [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)
- [COCO](https://cocodataset.org/#download)

# Data Preprocessing
```
# For Flickr30K
cd datasets
python split_flickr_data.py

# For COCO
cd datasets
python process_coco.py
```

# MACD Pretraining
```
python macd_pretraining.py --cfg cfg/pretrain-flickr-resnet.yml
```

# Unsupervised NLI
```
# For STS-B
python unsupervised_nli.py --cfg cfg/unsupervised/sts-b.yml
# For SNLI
python unsupervised_nli.py --cfg cfg/unsupervised/snli.yml
python snli_unsupervised.py --data_folder /home/a/MACD/unsupervised/flickr-resnet/snli
```

# Fine-tuning
```
# For STS-B
python run_finetuning.py --cfg cfg/finetune/sts-b.yml
# For SNLI
python run_finetuning.py --cfg cfg/finetune/snli.yml
```

# Citation
```
@inproceedings{cui2020unsupervised,
  title={Unsupervised Natural Language Inference via Decoupled Multimodal Contrastive Learning},
  author={Cui, Wanyun and Zheng, Guangyu and Wang, Wei},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={5511--5520},
  year={2020}
}
```
