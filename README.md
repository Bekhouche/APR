# APR: Adaptive Passage Retrieval for Arabic Text

This repository contains the implementation of **Adaptive Passage Retrieval (APR)**, a novel approach for Arabic text retrieval that combines a dual-encoder architecture with Attentive Relevance Scoring (ARS) for enhanced semantic matching.

## 📋 Requirements

```bash
torch>=1.9.0
transformers>=4.20.0
numpy>=1.21.0
scann>=1.2.6
```

## 🚀 Quick Start

```python
from apr import DynamicDPR, AttentiveRelevanceScoring
from transformers import AutoTokenizer

# Initialize model
model_name = "asafaya/bert-mini-arabic"
model = DynamicDPR(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.tokenizer = tokenizer

# Example documents
documents = [
    ("doc1", "هذا نص عربي يتحدث عن الذكاء الاصطناعي"),
    ("doc2", "البحث في استرجاع المعلومات مهم جداً"),
    ("doc3", "النماذج اللغوية تحدث ثورة في معالجة اللغة")
]

# Create index
model.create_index(documents, batch_size=32)

# Search
query = "الذكاء الاصطناعي"
results = model.search(query, top_k=5)
print(results)
```

## 🎯 Coming Soon

- **Training Scripts**: Complete training pipeline with data loaders
- **Inference Scripts**: Batch processing and evaluation utilities
- **Pre-trained Weights**: Models trained on Arabic datasets
- **Evaluation Benchmarks**: Performance on Arabic QA datasets
- **Complete Loss Implementation**: Full L_total with contrastive component

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{bekhouche2025enhanced,
    title={Enhanced Arabic Text Retrieval with Attentive Relevance Scoring},
    author={Bekhouche, Salah Eddine and Benlamoudi, Azeddine and Bounab, Yazid and Dornaika, Fadi and Hadid, Abdenour},
    booktitle={2025 IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
    year={2025},
    month={Aug--Sep},
    address={Istanbul, Turkey},
    organization={IEEE},
    note={Implementation available at: https://github.com/Bekhouche/APR}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
