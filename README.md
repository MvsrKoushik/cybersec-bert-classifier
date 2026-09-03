# Cybersecurity BERT Classifier

Merged and cleaned from the CyberSecBERT and scalable-baseline Colab experiments. The default path is a memory-efficient hashing-vectorizer/SGD classifier; transformer fine-tuning and BERTopic exploration are optional extras.

```bash
pip install -e .[dev]
pytest
```

Use time- or source-aware validation when training on NVD descriptions to avoid near-duplicate leakage. Raw NVD downloads and trained weights are excluded.

