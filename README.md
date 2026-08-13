# Financial Sentiment Analysis via Transfer Learning with Bi-LSTM

A PyTorch implementation of a transfer learning framework for domain-specific sentiment analysis. The pipeline pre-trains a Bidirectional Long Short-Term Memory (Bi-LSTM) model on general movie reviews (IMDb) and fine-tunes it on specialized financial text (`Financial PhraseBank`) using progressive unfreezing and differential learning rates.

---

## Project Overview

Sentiment analysis in financial contexts presents distinct challenges due to domain-specific terminology, specialized phrasing, and severe class imbalance (predominantly neutral phrasing). Standard general-purpose sentiment models often fail to capture financial nuance without domain adaptation.

This repository implements a 2-stage transfer learning pipeline:
1. **General Pre-Training**: Pre-trains a deep Bi-LSTM feature extractor on the IMDb dataset (2-class: positive vs. negative) to learn general linguistic and syntactic structures.
2. **Domain-Specific Fine-Tuning**: Adapts the pre-trained encoder to the Financial PhraseBank dataset (3-class: negative, neutral, positive) using a two-stage strategy (MLP warm-up followed by selective end-to-end fine-tuning).

### Key Architectural Highlights
- **Joint Vocabulary Alignment**: Constructs a shared vocabulary across both IMDb and Financial PhraseBank datasets prior to pre-training to preserve word index consistency across domain transfer.
- **Masked Mean Pooling**: Aggregates variable-length LSTM hidden state outputs across sequence steps while explicitly masking out `<PAD>` tokens.
- **Progressive Unfreezing & Differential Learning Rates**:
  - *Stage 1 (Warm-Up)*: Encoder weights (Embedding & Bi-LSTM) are frozen; only the newly initialized 3-class MLP classifier head is trained.
  - *Stage 2 (Fine-Tuning)*: The Bi-LSTM encoder is unfrozen and trained at a lower learning rate ($10^{-4}$) relative to the MLP head ($10^{-3}$), protecting pre-trained features from catastrophic forgetting.
- **Class-Weighted Cross-Entropy**: Employs square-root weighted loss balancing to compensate for neutral class dominance in financial phrases.

---

## Repository Structure

```text
.
├── README.md              # Project documentation and usage guide
├── requirements.txt       # Python environment dependencies
├── presentation/
│   └── presentation.tex   # Beamer slide deck presentation (LaTeX)
├── report/
│   └── report.tex         # Technical research report (LaTeX)
└── src/
    ├── config.py          # Configuration parameters, paths, and hyperparameters
    ├── data_prep.py        # Text cleaning, vocabulary building, and DataLoaders
    ├── engine.py          # PyTorch training, validation, and evaluation loops
    ├── model.py           # SentimentLSTM network definition with Masked Mean Pooling
    ├── main_pretrain.py   # Script for Stage 1 IMDb pre-training
    └── main_finetune.py   # Script for Stage 2 Financial PhraseBank fine-tuning
```

### Module Descriptions (`src/`)
- `config.py`: Defines global hardware device configuration, sequence parameters, vocabulary constraints, and hyperparameter constants.
- `data_prep.py`: Handles regex text cleaning, vocabulary generation with `<PAD>`/`<UNK>` token management, PyTorch `Dataset` wrappers, and dataset splitting.
- `model.py`: Implements `SentimentLSTM`, containing token embedding, a multi-layer Bi-LSTM, masked mean pooling, and a multi-layer perceptron (MLP) classification head.
- `engine.py`: Encapsulates modular `train_epoch()` and `evaluate()` functions, computing loss and accuracy metrics across mini-batches.
- `main_pretrain.py`: Trains the binary sentiment model on IMDb and saves the top-performing model weights.
- `main_finetune.py`: Loads IMDb pre-trained weights, adapts the MLP output dimension to 3 classes, executes classifier warm-up, and performs full selective fine-tuning.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- PyTorch 2.0+ (CUDA recommended for GPU acceleration)

### 2. Environment Setup
Clone the repository and create a virtual environment:

```bash
# Clone the repository
git clone https://github.com/17marche/Fondamenti-AI-Lab.git
cd Fondamenti-AI-Lab

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Dataset Placement
Ensure the raw datasets are placed in the `data/` directory:
- `data/imdb-dataset.csv`: IMDb Dataset of 50k Movie Reviews.
- `data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt`: Financial PhraseBank dataset (50% agreement subset).

---

## Pipeline Execution

### Step 1: Pre-Train on IMDb
Execute `main_pretrain.py` to train the initial Bi-LSTM model on movie reviews:

```bash
python src/main_pretrain.py
```

The script cleans the text, builds the shared vocabulary, trains the binary classification model for 5 epochs using `BCEWithLogitsLoss`, and saves the optimal model weights to `saved_models/sentiment_model.pth`.

### Step 2: Fine-Tune on Financial PhraseBank
Execute `main_finetune.py` to adapt the pre-trained model to financial sentiment:

```bash
python src/main_finetune.py
```

The fine-tuning pipeline automatically:
1. Loads the pre-trained weights from `saved_models/sentiment_model.pth`.
2. Replaces the 1-output linear layer with a 3-class classification head (`Negative`, `Neutral`, `Positive`).
3. Evaluates zero-shot baseline performance.
4. Executes **Stage 1**: Freezes encoder layers and trains the MLP classifier head (10 epochs warm-up).
5. Executes **Stage 2**: Unfreezes Bi-LSTM encoder layers and performs end-to-end fine-tuning with differential learning rates and early stopping (patience = 7).
6. Saves the final fine-tuned model checkpoint to `saved_models/sentiment_model_final_best.pth`.

---

## Documentation & Reports
- **Technical Report**: Found under [`report/report.tex`], detailing dataset statistics, architectural choices, and empirical validation results.
- **Presentation Deck**: Found under [`presentation/presentation.tex`], containing slide materials for academic presentation.