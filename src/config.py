import torch
import os

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models', 'sentiment_model.pth')

IMDB_CSV = os.path.join(DATA_DIR, 'imdb-dataset.csv')
FINANCIAL_PHRASEBANK_DIR = os.path.join(DATA_DIR, 'FinancialPhraseBank-v1.0')
FINANCIAL_DATA = os.path.join(FINANCIAL_PHRASEBANK_DIR, 'Sentences_50Agree.txt')

# Data Preprocessing
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 256
PAD_IDX = 0
UNK_IDX = 1

# Model Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.5

# Training Hyperparameters
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 5
WARMUP_EPOCHS = 10
FINETUNE_EPOCHS = 30
LEARNING_RATE_PRETRAIN = 1e-3

# Classes
FINANCIAL_CLASSES = 3 # negative, neutral, positive
