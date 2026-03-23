# Detecting Conspiracy Theory Against COVID-19 Vaccines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2211.13003-b31b1b.svg)](https://arxiv.org/abs/2211.13003)
[![Dataset on HuggingFace](https://img.shields.io/badge/🤗%20Dataset-HuggingFace-orange)](https://huggingface.co/datasets/AminHasibul/covid-vaccine-conspiracy)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AminHasibul/ConspiracyAgaintstCovidVaccines)
[![Cited](https://img.shields.io/badge/Citations-External%20Reuse-brightgreen)](https://arxiv.org/abs/2211.13003)

> **Paper:** Amin, M. H., Madanu, H., Lavu, S., Mansourifar, H., Alsagheer, D., & Shi, W. (2022).
> *"Detecting Conspiracy Theory Against COVID-19 Vaccines."* arXiv:2211.13003.
> University of Houston, Department of Computer Science.

---

## Overview

This repository provides a **manually labeled dataset and NLP classification pipeline** for detecting
conspiracy theories related to COVID-19 vaccines in social media text.

With vaccine hesitancy driven by online misinformation remaining a global public health challenge,
this work provides researchers, platform moderators, and policymakers with tools to study and
counter conspiracy theory spread at scale.

**This repository contains:**
- 📊 **598 manually labeled social media comments** — collected from North American online news portals and Facebook pages
- 🤖 **BERT-based and Perspective API classification pipelines** with full benchmark comparisons
- 📓 **Reproducible notebooks** for training, evaluation, and exploratory data analysis
- 🔍 **6-model comparison** across two paradigms (BERT embeddings vs. Google Perspective API)
- 🗄️ **Dataset on Hugging Face** for easy integration into research workflows

---

## Model Performance

All models evaluated using **10-fold cross-validation**.

### BERT Model Results

| Classifier | Accuracy | F1-Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| **BERT + Logistic Regression** | **69%** | **68%** | **67%** | **68%** |
| BERT + XGBoost | 66% | 66% | 67% | 65% |
| BERT + Gaussian Naïve Bayes | 51% | 51% | 52% | 51% |

### Google Perspective API Results

| Classifier | Accuracy | F1-Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| **Perspective + Gaussian NB** | **75%** | **75%** | **75%** | **75%** |
| Perspective + XGBoost | 65% | 63% | 65% | 65% |
| Perspective + Logistic Regression | 55% | 53% | 55% | 55% |

> **Key finding:** Perspective API + Gaussian Naïve Bayes achieves the highest accuracy (75%).
> BERT + Logistic Regression is the best pure neural approach (69%).
> An 8–9% performance increase was observed when data volume was increased from 400 to 598 samples,
> suggesting further improvement is expected with larger datasets.

---

## Dataset

### Primary Dataset: `finaldataset y.csv`

| Property | Details |
|----------|---------|
| Total samples | 598 unique comments (after deduplication from 950 collected) |
| Source | Online news portals and Facebook pages (North American users) |
| Language | English only |
| Annotation | Manual binary labeling by research team |
| Collection period | 2021–2022 (COVID-19 vaccine rollout) |
| Privacy | No personal information (name, location, gender) included |
| Format | CSV |
| License | MIT |

**Label Schema:**

| Label | `conspiracy_found` | Meaning |
|-------|--------------------|---------|
| `1` | Yes | Comment contains a conspiracy theory about COVID-19 vaccines |
| `0` | No | Comment is neutral or in favor of vaccination |

**Class Distribution:** Approximately balanced (equal positive/negative samples — see Figure 2 in paper).

**Sample entries from the paper:**

| Comment | Label |
|---------|-------|
| "After getting vaccine you catch heart diseases" | 1 (Yes) |
| "Vaccination can have an impact on gender change" | 1 (Yes) |
| "Fully vaccination can reduce death rate for COVID-19" | 0 (No) |
| "After getting one dose of the J & J vaccine to boost the immune system" | 0 (No) |

**Dataset also available on Hugging Face:**
```python
from datasets import load_dataset
ds = load_dataset("AminHasibul/covid-vaccine-conspiracy")
print(ds["train"][0])
```

---

## Quick Start

### Option 1: Hugging Face (Recommended)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load dataset
ds = load_dataset("AminHasibul/covid-vaccine-conspiracy")

# Fine-tune with HuggingFace Trainer — see Hugging Face dataset card for full training code
```

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/AminHasibul/ConspiracyAgaintstCovidVaccines.git
cd ConspiracyAgaintstCovidVaccines

# Install dependencies (modern HuggingFace stack)
pip install transformers datasets torch pandas scikit-learn matplotlib seaborn

# Open notebook
jupyter notebook Bert_Covid_Cons.ipynb
```

> **Note on dependencies:** The original paper used `bert-as-service` with TensorFlow 1.x.
> For reproducibility with modern tooling, we recommend using `transformers` (HuggingFace).
> The Hugging Face dataset card provides updated training code compatible with Python 3.8+.

---

## Repository Structure

```
ConspiracyAgaintstCovidVaccines/
├── README.md                        # This file
├── DOCUMENTATION.md                 # Extended methodology documentation
├── LICENSE                          # MIT License
├── Bert_Covid_Cons.ipynb           # BERT classification and evaluation pipeline
├── Data_analysis_using_BERT.ipynb  # EDA, word frequency, embedding visualization
├── finaldataset y.csv              # Main labeled dataset (598 samples)
├── finaldataset.csv                # Word frequency analysis data
└── .gitignore
```

---

## Notebooks

### `Bert_Covid_Cons.ipynb` — Classification Pipeline
- BERT-Base Uncased fine-tuning (12-layer, 768-hidden, 12-heads, 110M parameters)
- Logistic Regression, XGBoost, and Gaussian Naïve Bayes classifiers on BERT embeddings
- Google Perspective API integration for toxicity scoring
- 10-fold cross-validation evaluation
- Full metrics: accuracy, F1, precision, recall, confusion matrix

### `Data_analysis_using_BERT.ipynb` — Exploratory Data Analysis
- Label distribution analysis (Figure 2 in paper)
- Most common word frequency visualization (Figure 1 in paper)
- Text preprocessing: stop word removal, abbreviation normalization, lowercasing
- BERT embedding visualization and clustering

---

## Research Context

### Problem Statement

COVID-19 conspiracy theories — including claims about 5G networks, Bill Gates microchips,
Chinese bioweapons, and vaccine-induced health conditions — caused measurable real-world harm
including attacks on cell towers, violence against Asian-Americans, and widespread vaccine hesitancy.

Automated detection of these narratives at scale is a critical NLP challenge at the intersection
of public health and computational social science.

### Contributions

1. A manually labeled dataset of vaccine-related social media comments for conspiracy detection
2. Comparative evaluation of BERT embeddings vs. Google Perspective API across three classifiers
3. Evidence that data volume directly improves classification performance (8–9% gain from 400→598 samples)
4. Baseline results for future research on low-resource health misinformation detection

### Limitations (from paper)

- Data sourced from North American users — may not generalize to other regions
- English-only — does not cover multilingual vaccine misinformation
- Dataset size limits statistical generalization
- User demographic information (age, gender) not collected

### Related Work from the Same Group

- [Multilingual Financial Fraud Detection — IEEE CICN 2026](https://arxiv.org/abs/2603.11358) — extends NLP
  misinformation detection methodology to financial fraud in Bangla-English code-mixed text
- [Continual Learning for Adaptive AI Systems — arXiv 2025](https://arxiv.org/abs/2510.07648) — addresses
  model adaptation and catastrophic forgetting in production ML systems

---

## Ethical Statement

This dataset and code are released **strictly for research and educational purposes**.

- Content is provided to **detect and counter** misinformation — not to spread it
- No personal information about commenters is included in the dataset
- All source comments were publicly posted; collection complied with platform terms
- Researchers are encouraged to review the Hugging Face dataset card for full ethical guidelines
- Do not use this pipeline to unfairly target, label, or penalize individuals or communities

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{amin2022detecting,
  title={Detecting conspiracy theory against covid-19 vaccines},
  author={Amin, Md Hasibul and Madanu, Harika and Lavu, Sahithi and Mansourifar, Hadi and Alsagheer, Dana and Shi, Weidong},
  journal={arXiv preprint arXiv:2211.13003},
  year={2022}
}
```

**arXiv:** https://arxiv.org/abs/2211.13003

---

## Authors

**Md Hasibul Amin** (First Author) — [GitHub](https://github.com/AminHasibul) | [LinkedIn](https://linkedin.com/in/aminhasibul) | [Google Scholar](https://scholar.google.com)
Department of Computer Science, University of Houston

Harika Madanu · Sahithi Lavu · Hadi Mansourifar · Dana Alsagheer · Weidong Shi
Department of Computer Science, University of Houston

*Data collection supported by COSC 6376 Cloud Computing Course (Fall 2021), University of Houston.*

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*For questions or issues, please [open an issue](https://github.com/AminHasibul/ConspiracyAgaintstCovidVaccines/issues).*
