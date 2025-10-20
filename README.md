
# Conspiracy Against COVID-19 Vaccines Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AminHasibul/ConspiracyAgaintstCovidVaccines)

## 📖 Overview

This repository provides a comprehensive dataset and machine learning pipeline for detecting conspiracy theories related to COVID-19 vaccines. The project uses **BERT (Bidirectional Encoder Representations from Transformers)** to analyze and classify text content for conspiratorial narratives.

The repository includes:
- 🗄️ **Labeled dataset** of COVID-19 vaccine-related comments
- 🤖 **BERT-based classification model** implementation
- 📊 **Data analysis notebooks** for exploratory analysis
- 🔍 **Pre-processing and feature extraction** pipelines

## 🎯 Key Features

- **State-of-the-art NLP**: Utilizes BERT embeddings for text representation
- **Labeled Dataset**: 600+ manually labeled comments with conspiracy theory annotations
- **End-to-end Pipeline**: Complete workflow from data loading to model evaluation
- **Google Colab Ready**: Notebooks designed to run seamlessly on Google Colab
- **Reproducible Research**: Well-documented code for academic and research purposes

## 📊 Dataset Description

The repository contains two primary dataset files:

### 1. `finaldataset y.csv` (Main Dataset)
- **Size**: 616 labeled samples
- **Columns**:
  - `comments`: Text content of COVID-19 vaccine-related comments
  - `label`: Binary label (1 = conspiracy theory detected, 0 = no conspiracy)
  - `conspiracy_found`: Human-readable label ("Yes" or "No")

**Example entries**:
```csv
comments,label,conspiracy_found
"Vaccine is a failure",1,Yes
"My mom tested covid positive after getting vaccinated",1,Yes
```

### 2. `finaldataset.csv` (Word Frequency Data)
- Contains common word frequencies from the dataset
- Useful for exploratory data analysis and feature engineering
- **Columns**: `common_word`, `Count`

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- TensorFlow 1.14 (for BERT serving)
- Google Colab account (recommended) or local Jupyter environment

### Installation

#### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge above
2. Select either notebook:
   - `Bert_Covid_Cons.ipynb` - BERT model training and evaluation
   - `Data_analysis_using_BERT.ipynb` - Exploratory data analysis
3. Run cells sequentially

#### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/AminHasibul/ConspiracyAgainststCovidVaccines.git
cd ConspiracyAgainststCovidVaccines

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install tensorflow==1.14
pip install bert-serving-server
pip install bert-serving-client
pip install nltk gensim numpy pandas scikit-learn matplotlib seaborn

# Download BERT pre-trained model
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

## 📓 Notebooks

### 1. `Bert_Covid_Cons.ipynb`
**Purpose**: Main model training and classification pipeline

**Key Steps**:
1. Download and setup BERT pre-trained model (uncased L-12 H-768 A-12)
2. Initialize BERT serving server
3. Load and preprocess the dataset
4. Generate BERT embeddings for text
5. Train classification model
6. Evaluate model performance

**Technologies Used**:
- BERT (bert-serving-server/client)
- TensorFlow 1.14
- NLTK for text preprocessing
- NumPy for numerical operations

### 2. `Data_analysis_using_BERT.ipynb`
**Purpose**: Exploratory data analysis and visualization

**Key Steps**:
1. Load dataset and perform initial exploration
2. Analyze text characteristics and patterns
3. Visualize word frequencies and distributions
4. Generate BERT embeddings for analysis
5. Perform dimensionality reduction and clustering
6. Create visualizations of conspiracy vs. non-conspiracy patterns

## 🔧 Usage

### Running the BERT Classification Pipeline

```python
# Start BERT serving (run in terminal)
bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12

# In Python script or notebook
from bert_serving.client import BertClient
import pandas as pd

# Initialize BERT client
bc = BertClient()

# Load your data
df = pd.read_csv('finaldataset y.csv')

# Get embeddings
embeddings = bc.encode(df['comments'].tolist())

# Use embeddings for classification
# (See notebooks for complete implementation)
```

### Analyzing New Text

```python
# Load trained model (after running notebook)
# Classify new text
new_text = ["I don't trust the vaccine because it was developed too quickly"]
embedding = bc.encode(new_text)
prediction = model.predict(embedding)  # model from notebook

print(f"Conspiracy theory detected: {prediction[0] == 1}")
```

## 📈 Model Performance

The BERT-based classification model achieves strong performance in detecting conspiracy theories. Detailed metrics and evaluation results are available in the notebooks.

**Key Metrics** (see notebooks for complete results):
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🗂️ Repository Structure

```
ConspiracyAgainststCovidVaccines/
├── README.md                          # This file
├── DOCUMENTATION.md                   # Detailed documentation
├── LICENSE                            # MIT License
├── Bert_Covid_Cons.ipynb             # Main BERT classification notebook
├── Data_analysis_using_BERT.ipynb    # Data analysis notebook
├── finaldataset y.csv                # Main labeled dataset
├── finaldataset.csv                  # Word frequency data
└── .gitignore                        # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the code, add features, or enhance documentation:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this dataset or code in your research, please cite the following paper:

```bibtex
@article{amin2022detecting,
  title={Detecting conspiracy theory against covid-19 vaccines},
  author={Amin, Md Hasibul and Madanu, Harika and Lavu, Sahithi and Mansourifar, Hadi and Alsagheer, Dana and Shi, Weidong},
  journal={arXiv preprint arXiv:2211.13003},
  year={2022}
}
```

**ArXiv Link**: [https://arxiv.org/abs/2211.13003](https://arxiv.org/abs/2211.13003)

## 👥 Authors

- **Md Hasibul Amin** - [GitHub](https://github.com/AminHasibul)
- Harika Madanu
- Sahithi Lavu
- Hadi Mansourifar
- Dana Alsagheer
- Weidong Shi

## 🔗 Related Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [bert-as-service](https://github.com/hanxiao/bert-as-service)
- [TensorFlow](https://www.tensorflow.org/)

## ❓ Support

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/AminHasibul/ConspiracyAgainststCovidVaccines/issues)
- Contact the authors through GitHub

## 🙏 Acknowledgments

- Google Research for BERT pre-trained models
- The research community for COVID-19 misinformation detection
- Contributors and users of this repository

---

**Note**: This repository is for research and educational purposes. The dataset contains real social media content and should be used responsibly
