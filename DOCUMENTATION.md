# Detailed Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Dataset Details](#dataset-details)
5. [Code Architecture](#code-architecture)
6. [Workflow Explanation](#workflow-explanation)
7. [BERT Model Details](#bert-model-details)
8. [Notebook Walkthroughs](#notebook-walkthroughs)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)
12. [Extending the Project](#extending-the-project)

---

## Introduction

This project implements a machine learning pipeline for detecting conspiracy theories in COVID-19 vaccine-related text using BERT (Bidirectional Encoder Representations from Transformers). The system analyzes natural language text and classifies it as containing conspiracy narratives or not.

### Research Context

The COVID-19 pandemic led to widespread misinformation and conspiracy theories about vaccines. This project addresses the need for automated detection of such content to:
- Monitor public sentiment
- Identify misinformation patterns
- Support fact-checking efforts
- Understand conspiracy theory characteristics

### Technical Approach

The project uses:
- **Pre-trained BERT**: Transfer learning from Google's BERT model
- **Fine-tuning**: Adaptation to conspiracy theory detection task
- **Text embeddings**: Dense vector representations of text
- **Classification**: Binary classification (conspiracy vs. non-conspiracy)

---

## System Requirements

### Hardware Requirements

**Minimum**:
- CPU: 4 cores (Intel Core i5 or equivalent)
- RAM: 8 GB
- Storage: 5 GB free space
- Network: Internet connection for downloading models

**Recommended**:
- CPU: 8+ cores (Intel Core i7 or equivalent)
- RAM: 16 GB or higher
- GPU: NVIDIA GPU with CUDA support (for faster processing)
- Storage: 10 GB free space
- Network: High-speed internet connection

**Google Colab** (Recommended):
- Free tier: Sufficient for running notebooks
- Pro tier: Recommended for faster processing and longer sessions

### Software Requirements

**Required**:
- Python 3.7, 3.8, or 3.9
- TensorFlow 1.14 (specific version required for BERT serving)
- pip (Python package manager)
- Git (for cloning repository)

**Supported Operating Systems**:
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS 10.14+
- Windows 10+ (with WSL2 recommended)
- Google Colab (browser-based)

---

## Installation Guide

### Option 1: Google Colab (Easiest)

Google Colab provides a pre-configured environment with most dependencies installed.

1. **Open Notebook in Colab**:
   - Navigate to [Google Colab](https://colab.research.google.com/)
   - Go to File → Open notebook → GitHub
   - Enter: `AminHasibul/ConspiracyAgainststCovidVaccines`
   - Select desired notebook

2. **Run Setup Cells**:
   - Execute cells sequentially from top to bottom
   - BERT model download: ~400 MB, takes 2-5 minutes
   - Installation cells will handle all dependencies

3. **Upload Dataset**:
   - If not using GitHub directly, upload CSV files to Colab
   - Use Files panel on left sidebar
   - Drag and drop `finaldataset y.csv`

### Option 2: Local Installation (Linux/macOS)

#### Step 1: Clone Repository
```bash
git clone https://github.com/AminHasibul/ConspiracyAgainststCovidVaccines.git
cd ConspiracyAgainststCovidVaccines
```

#### Step 2: Create Virtual Environment
```bash
# Using venv (recommended)
python3 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n conspiracy-detection python=3.7
conda activate conspiracy-detection
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install TensorFlow 1.14 (critical version)
pip install tensorflow==1.14

# Install BERT serving
pip install bert-serving-server
pip install bert-serving-client

# Install NLP and data science libraries
pip install nltk==3.5
pip install gensim==3.8.3
pip install numpy==1.19.5
pip install pandas==1.1.5
pip install scikit-learn==0.24.2
pip install matplotlib==3.3.4
pip install seaborn==0.11.2

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Step 4: Download BERT Model
```bash
# Download pre-trained BERT (uncased, 12-layer, 768-hidden)
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

# Extract
unzip uncased_L-12_H-768_A-12.zip

# Verify extraction
ls uncased_L-12_H-768_A-12/
# Should show: bert_config.json, bert_model.ckpt.*, vocab.txt
```

#### Step 5: Start BERT Serving
```bash
# Start BERT server (in separate terminal or background)
bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12 -num_worker=2

# Or in background
nohup bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12 > bert_server.log 2>&1 &
```

#### Step 6: Launch Jupyter
```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch notebook
jupyter notebook

# Open either notebook in browser
```

### Option 3: Local Installation (Windows)

#### Prerequisites
- Install [Python 3.7+](https://www.python.org/downloads/)
- Install [Git for Windows](https://git-scm.com/download/win)
- Optional: Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install) for better compatibility

#### Using WSL2 (Recommended)
```bash
# Follow Linux installation steps in WSL2 terminal
```

#### Native Windows (Alternative)
```cmd
# Clone repository
git clone https://github.com/AminHasibul/ConspiracyAgainststCovidVaccines.git
cd ConspiracyAgainststCovidVaccines

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages
pip install tensorflow==1.14
pip install bert-serving-server bert-serving-client
pip install nltk gensim numpy pandas scikit-learn matplotlib seaborn

# Download BERT model (use browser or wget for Windows)
# Extract zip file manually

# Start BERT server
bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12

# Launch Jupyter in separate command window
jupyter notebook
```

### Verification

Test your installation:
```python
# Test BERT client connection
from bert_serving.client import BertClient
bc = BertClient()
print(bc.encode(["Hello, World!"]).shape)
# Should print: (1, 768)

# Test imports
import tensorflow as tf
import pandas as pd
import nltk
import numpy as np

print("TensorFlow version:", tf.__version__)
print("Installation successful!")
```

---

## Dataset Details

### Dataset Collection Methodology

The dataset was collected and curated through:
1. **Source**: Social media platforms (primarily Twitter)
2. **Keywords**: COVID-19, vaccine, vaccination, immunization
3. **Time period**: 2020-2022 (pandemic period)
4. **Language**: English
5. **Annotation**: Manual labeling by human annotators

### Data Labeling Guidelines

Comments were labeled as conspiracy theories if they contained:
- Unsubstantiated claims about vaccine dangers
- References to hidden agendas or cover-ups
- Distrust based on non-scientific reasoning
- Promotion of alternative explanations without evidence
- Anti-establishment narratives related to vaccines

### Dataset Statistics

#### finaldataset y.csv
```
Total Samples: 616
Conspiracy (label=1): ~300 samples (~48.7%)
Non-conspiracy (label=0): ~316 samples (~51.3%)
Average comment length: ~50-150 characters
Language: English
Format: CSV with UTF-8 encoding
```

#### Column Descriptions

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `comments` | String | Text content of comment | "Vaccine is dangerous" |
| `label` | Integer | Binary label (0 or 1) | 1 |
| `conspiracy_found` | String | Human-readable label | "Yes" |

### Data Quality

- **Cleaned**: Removed duplicates and low-quality samples
- **Balanced**: Near 50/50 split between classes
- **Validated**: Multiple annotator agreement
- **Anonymized**: Personal information removed

### Ethical Considerations

- Data collected from public sources
- Personal identifiers removed
- Used for research purposes only
- Respects platform terms of service
- No individual profiling intended

---

## Code Architecture

### Project Structure

```
ConspiracyAgainststCovidVaccines/
│
├── Notebooks/
│   ├── Bert_Covid_Cons.ipynb              # Main classification pipeline
│   └── Data_analysis_using_BERT.ipynb     # EDA and analysis
│
├── Data/
│   ├── finaldataset y.csv                 # Main labeled dataset
│   └── finaldataset.csv                   # Word frequency data
│
├── Models/
│   └── uncased_L-12_H-768_A-12/          # BERT pre-trained model
│       ├── bert_config.json
│       ├── bert_model.ckpt.*
│       └── vocab.txt
│
└── Documentation/
    ├── README.md                          # Main readme
    └── DOCUMENTATION.md                   # This file
```

### Component Overview

#### 1. Data Loading Module
- **Purpose**: Load and validate datasets
- **Functions**: CSV parsing, data validation
- **Dependencies**: pandas

#### 2. Preprocessing Module
- **Purpose**: Clean and prepare text data
- **Functions**: 
  - Tokenization (NLTK word_tokenize)
  - Lowercasing
  - Stop word removal
  - Stemming/Lemmatization
- **Dependencies**: NLTK, re

#### 3. BERT Embedding Module
- **Purpose**: Generate text embeddings
- **Functions**: 
  - BERT client initialization
  - Batch encoding
  - Embedding extraction
- **Dependencies**: bert-serving-client

#### 4. Model Training Module
- **Purpose**: Train classification models
- **Functions**:
  - Data splitting
  - Model training
  - Hyperparameter tuning
- **Dependencies**: scikit-learn, TensorFlow

#### 5. Evaluation Module
- **Purpose**: Assess model performance
- **Functions**:
  - Metric calculation
  - Confusion matrix
  - Visualization
- **Dependencies**: scikit-learn, matplotlib

---

## Workflow Explanation

### End-to-End Pipeline

```
1. Data Loading
   ↓
2. Text Preprocessing
   ↓
3. BERT Server Initialization
   ↓
4. Text Encoding (BERT Embeddings)
   ↓
5. Feature Extraction
   ↓
6. Model Training
   ↓
7. Model Evaluation
   ↓
8. Prediction on New Data
```

### Detailed Steps

#### Step 1: Data Loading
```python
import pandas as pd

# Load dataset
df = pd.read_csv('finaldataset y.csv')

# Inspect data
print(df.head())
print(df.info())
print(df['label'].value_counts())
```

#### Step 2: Text Preprocessing
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_comments'] = df['comments'].apply(preprocess_text)
```

#### Step 3: BERT Server Initialization
```python
from bert_serving.client import BertClient

# Connect to BERT server
bc = BertClient()

# Test connection
test_embedding = bc.encode(["test sentence"])
print(f"Embedding shape: {test_embedding.shape}")  # (1, 768)
```

#### Step 4: Generate Embeddings
```python
# Generate embeddings for all comments
comments_list = df['cleaned_comments'].tolist()

# Encode in batches for efficiency
embeddings = bc.encode(comments_list)

print(f"Embeddings shape: {embeddings.shape}")  # (616, 768)
```

#### Step 5: Train/Test Split
```python
from sklearn.model_selection import train_test_split

X = embeddings
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

#### Step 6: Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Try multiple classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf', probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
```

#### Step 7: Model Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Best model evaluation
best_model = models['Logistic Regression']  # Example
y_pred = best_model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred, 
                          target_names=['Non-conspiracy', 'Conspiracy']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-conspiracy', 'Conspiracy'],
            yticklabels=['Non-conspiracy', 'Conspiracy'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
```

#### Step 8: Prediction on New Data
```python
def predict_conspiracy(text):
    # Preprocess
    cleaned = preprocess_text(text)
    
    # Generate embedding
    embedding = bc.encode([cleaned])
    
    # Predict
    prediction = best_model.predict(embedding)[0]
    probability = best_model.predict_proba(embedding)[0]
    
    return {
        'text': text,
        'prediction': 'Conspiracy' if prediction == 1 else 'Non-conspiracy',
        'confidence': max(probability)
    }

# Example usage
result = predict_conspiracy("I don't trust vaccines made so quickly")
print(result)
```

---

## BERT Model Details

### BERT Architecture

**Model Used**: `BERT-Base, Uncased`
- **Layers**: 12 transformer encoder layers
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Parameters**: ~110 million
- **Vocabulary**: 30,522 WordPiece tokens

### Why BERT?

1. **Bidirectional Context**: Understands context from both directions
2. **Transfer Learning**: Pre-trained on large corpus (Wikipedia + BookCorpus)
3. **State-of-the-art**: Achieves high performance on NLP tasks
4. **Semantic Understanding**: Captures nuanced meanings

### BERT Embeddings

- **Output**: 768-dimensional dense vector per input
- **Representation**: Contextual word embeddings
- **Pooling**: [CLS] token used for sentence-level representation
- **Fixed Length**: All sentences mapped to same dimensionality

### BERT Serving Configuration

```bash
bert-serving-start \
  -max_seq_len=128 \          # Maximum sequence length
  -model_dir=uncased_L-12_H-768_A-12 \  # Model directory
  -num_worker=2 \              # Number of worker processes
  -max_batch_size=256 \        # Maximum batch size
  -pooling_strategy=REDUCE_MEAN  # Embedding pooling strategy
```

**Parameters Explanation**:
- `max_seq_len`: Truncate/pad sequences to 128 tokens
- `num_worker`: Use 2 CPU workers for parallel processing
- `pooling_strategy`: Average token embeddings for sentence representation

---

## Notebook Walkthroughs

### Bert_Covid_Cons.ipynb

**Purpose**: Complete classification pipeline from data to predictions

**Cell-by-Cell Guide**:

1. **Cell 1-2**: Environment Setup
   - Download BERT model
   - Install virtualenv

2. **Cell 3-6**: Dependency Installation
   - TensorFlow 1.14
   - BERT serving packages
   - NLP libraries

3. **Cell 7-9**: BERT Server Setup
   - Extract BERT model
   - Start BERT server in background
   - Install BERT client

4. **Cell 10-12**: Client Initialization
   - Import BertClient
   - Test connection
   - Initialize token lists

5. **Cell 13-15**: Data Preprocessing
   - Import NLTK, gensim
   - Download NLTK data
   - Define preprocessing functions

6. **Cell 16-18**: Data Loading
   - Load CSV dataset
   - Explore data structure
   - Check label distribution

7. **Cell 19-21**: Text Cleaning
   - Apply preprocessing
   - Remove stopwords
   - Tokenize text

8. **Cell 22-24**: Embedding Generation
   - Encode comments using BERT
   - Store embeddings
   - Verify shapes

9. **Cell 25-26**: Model Training & Evaluation
   - Split data
   - Train classifiers
   - Evaluate performance
   - Visualize results

**Expected Runtime**: 15-30 minutes on Google Colab

### Data_analysis_using_BERT.ipynb

**Purpose**: Exploratory data analysis and visualization

**Key Sections**:

1. **Data Overview**
   - Load and inspect dataset
   - Statistical summaries
   - Missing value analysis

2. **Text Analysis**
   - Word frequency analysis
   - Common terms in conspiracy vs. non-conspiracy
   - Text length distribution

3. **Visualization**
   - Word clouds
   - Bar charts of common words
   - Label distribution

4. **BERT Embeddings Analysis**
   - Generate embeddings
   - Dimensionality reduction (t-SNE, PCA)
   - Clustering visualization

5. **Pattern Discovery**
   - Identify conspiracy markers
   - Analyze linguistic patterns
   - Feature importance

**Expected Runtime**: 10-20 minutes on Google Colab

---

## API Reference

### BertClient

```python
from bert_serving.client import BertClient

bc = BertClient()
```

**Methods**:

#### `encode(texts, is_tokenized=False, show_tokens=False)`
Generate embeddings for text(s).

**Parameters**:
- `texts` (List[str] or str): Input text(s)
- `is_tokenized` (bool): Whether input is pre-tokenized
- `show_tokens` (bool): Return tokens with embeddings

**Returns**:
- `numpy.ndarray`: Embeddings of shape (n_samples, 768)

**Example**:
```python
embeddings = bc.encode(["Hello world", "BERT is great"])
print(embeddings.shape)  # (2, 768)
```

#### `encode_async(texts, is_tokenized=False)`
Async version of encode for large batches.

### Preprocessing Functions

#### `preprocess_text(text)`
Clean and prepare text for BERT encoding.

**Parameters**:
- `text` (str): Raw input text

**Returns**:
- `str`: Cleaned text

**Operations**:
- Lowercasing
- URL removal
- Special character removal
- Stopword removal

### Model Training Functions

#### `train_classifier(X_train, y_train, model_type='logistic')`
Train classification model on BERT embeddings.

**Parameters**:
- `X_train` (ndarray): Training embeddings
- `y_train` (ndarray): Training labels
- `model_type` (str): 'logistic', 'random_forest', or 'svm'

**Returns**:
- Trained model object

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: BERT Server Won't Start

**Error**:
```
ERROR:tensorflow:Error recorded from training_loop: Model not found
```

**Solution**:
```bash
# Verify BERT model extraction
ls uncased_L-12_H-768_A-12/

# Should contain:
# - bert_config.json
# - bert_model.ckpt.data-00000-of-00001
# - bert_model.ckpt.index
# - bert_model.ckpt.meta
# - vocab.txt

# Re-download if files missing
rm -rf uncased_L-12_H-768_A-12*
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

#### Issue 2: TensorFlow Version Conflict

**Error**:
```
ImportError: cannot import name 'xxx' from 'tensorflow'
```

**Solution**:
```bash
# Uninstall existing TensorFlow
pip uninstall tensorflow tensorflow-gpu

# Install specific version
pip install tensorflow==1.14
```

#### Issue 3: BERT Client Connection Failed

**Error**:
```
TimeoutError: Server not ready after 10s
```

**Solution**:
```bash
# Check if server is running
ps aux | grep bert-serving-start

# Kill existing server
pkill -f bert-serving-start

# Restart with verbose logging
bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12 -verbose
```

#### Issue 4: Out of Memory (OOM)

**Error**:
```
MemoryError: Unable to allocate array
```

**Solution**:
```python
# Process in smaller batches
batch_size = 32
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = bc.encode(batch)
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)
```

#### Issue 5: NLTK Data Not Found

**Error**:
```
LookupError: Resource 'punkt' not found
```

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

#### Issue 6: Jupyter Notebook Kernel Crash

**Symptoms**: Kernel dies when running BERT encoding

**Solution**:
1. Reduce batch size
2. Increase RAM in Colab (Runtime → Change runtime type)
3. Restart kernel and clear outputs
4. Process data in chunks

### Performance Issues

#### Slow Encoding

**Problem**: BERT encoding takes too long

**Solutions**:
1. Increase workers:
```bash
bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12 -num_worker=4
```

2. Use GPU (if available):
```bash
bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12 -device_map -1 -gpu_memory_fraction 0.5
```

3. Reduce sequence length:
```bash
bert-serving-start -max_seq_len=64 -model_dir=uncased_L-12_H-768_A-12
```

### Data Issues

#### CSV Encoding Errors

**Error**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution**:
```python
# Try different encodings
df = pd.read_csv('finaldataset y.csv', encoding='latin-1')
# or
df = pd.read_csv('finaldataset y.csv', encoding='iso-8859-1')
```

---

## Performance Optimization

### Speed Improvements

1. **Batch Processing**:
```python
# Instead of one-by-one
for text in texts:
    embedding = bc.encode([text])

# Use batch encoding
embeddings = bc.encode(texts)  # Much faster
```

2. **Parallel Workers**:
```bash
# Use more workers for parallel processing
bert-serving-start -max_seq_len=128 -model_dir=uncased_L-12_H-768_A-12 -num_worker=4
```

3. **Caching Embeddings**:
```python
import pickle

# Save embeddings
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Load later
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
```

### Memory Optimization

1. **Process in Chunks**:
```python
def process_in_chunks(texts, chunk_size=100):
    embeddings_list = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        chunk_emb = bc.encode(chunk)
        embeddings_list.append(chunk_emb)
    return np.vstack(embeddings_list)
```

2. **Use Float16**:
```python
# Reduce memory by half
embeddings = bc.encode(texts).astype(np.float16)
```

### Model Optimization

1. **Feature Selection**:
```python
from sklearn.decomposition import PCA

# Reduce dimensionality
pca = PCA(n_components=128)
reduced_embeddings = pca.fit_transform(embeddings)
```

2. **Model Simplification**:
```python
# Use simpler models for faster inference
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log', max_iter=1000)
model.fit(X_train, y_train)
```

---

## Extending the Project

### Adding New Features

#### 1. Multi-class Classification
Extend beyond binary to classify types of conspiracy theories:

```python
# Update labels
conspiracy_types = {
    0: 'No conspiracy',
    1: 'Vaccine danger',
    2: 'Government control',
    3: 'Economic conspiracy',
    4: 'Medical distrust'
}

# Use multi-class classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial')
```

#### 2. Sentiment Analysis
Add sentiment scoring:

```python
from transformers import pipeline

sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']
```

#### 3. Topic Modeling
Discover conspiracy themes:

```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5)
topics = lda.fit_transform(embeddings)
```

#### 4. Real-time Detection
Create API endpoint:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    embedding = bc.encode([text])
    prediction = model.predict(embedding)[0]
    return jsonify({'conspiracy': bool(prediction)})

if __name__ == '__main__':
    app.run(port=5000)
```

### Integration Ideas

1. **Twitter Bot**: Monitor tweets in real-time
2. **Browser Extension**: Flag suspicious content
3. **Fact-checking Tool**: Support human fact-checkers
4. **Dashboard**: Visualize conspiracy trends
5. **Educational Tool**: Teach media literacy

### Research Directions

1. **Cross-lingual Detection**: Extend to other languages
2. **Temporal Analysis**: Track conspiracy evolution
3. **Network Analysis**: Study conspiracy spread patterns
4. **Multimodal Detection**: Include images and videos
5. **Explainability**: Understand model decisions

---

## Appendix

### Glossary

- **BERT**: Bidirectional Encoder Representations from Transformers
- **Embedding**: Dense vector representation of text
- **Transfer Learning**: Using pre-trained model for new task
- **Fine-tuning**: Adapting pre-trained model to specific task
- **Tokenization**: Splitting text into words/subwords
- **Classification**: Assigning labels to data

### Additional Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [bert-as-service GitHub](https://github.com/hanxiao/bert-as-service)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

### Version History

- **v1.0.0** (2022): Initial release
- **v1.1.0** (2025): Enhanced documentation

---

**Last Updated**: October 2025

For questions or contributions, please open an issue on GitHub.
