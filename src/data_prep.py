"""Data preprocessing utilities and PyTorch Dataset/DataLoader generators."""

import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from config import FINANCIAL_DATA, PAD_IDX, UNK_IDX


def clean_text(text: str) -> List[str]:
    """Cleans a raw text string and splits it into tokens.

    Converts input text to lowercase, strips HTML tags, removes special
    characters and punctuation, normalizes whitespaces, and splits text
    into word tokens.

    Args:
        text: Raw text string to be processed.

    Returns:
        List[str]: List of cleaned word tokens.
    """
    text = str(text).lower()

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-alphanumeric characters and punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize multiple consecutive spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize by whitespace
    tokens = text.split()

    return tokens


def build_vocab(cleaned_texts: List[List[str]], max_vocab_size: int) -> Dict[str, int]:
    """Constructs a word-to-index mapping vocabulary based on token frequency.

    Args:
        cleaned_texts: List of tokenized text sequences.
        max_vocab_size: Maximum vocabulary size (including special tokens).

    Returns:
        Dict[str, int]: Mapping dictionary from token strings to integer indices.
    """
    word_counts = Counter()
    for tokens in cleaned_texts:
        word_counts.update(tokens)

    # Reserve 2 slots for <PAD> and <UNK> special tokens
    most_common_words = word_counts.most_common(max_vocab_size - 2)

    word_to_idx = {
        "<PAD>": PAD_IDX,
        "<UNK>": UNK_IDX,
    }

    next_idx = max(PAD_IDX, UNK_IDX) + 1
    for word, _ in most_common_words:
        word_to_idx[word] = next_idx
        next_idx += 1

    return word_to_idx


class TextDataset(Dataset):
    """PyTorch Dataset for numericalizing and padding text sequences.

    Attributes:
        texts (List[List[str]]): List of tokenized input texts.
        labels (List[float] | List[int]): Targets for binary or multiclass classification.
        word_to_idx (Dict[str, int]): Vocabulary word-to-index dictionary.
        max_len (int): Maximum sequence length for padding/truncation.
    """

    def __init__(
        self,
        texts: List[List[str]],
        labels: List[float],
        word_to_idx: Dict[str, int],
        max_len: int,
    ):
        """Initializes TextDataset instance parameters."""
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self) -> int:
        """Returns total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves and numericalizes a single sample at the given index.

        Args:
            idx: Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor: LongTensor of shape `(max_len,)` with token indices.
                - y_tensor: FloatTensor containing target class label.
        """
        tokens = self.texts[idx]
        label = self.labels[idx]

        # Numericalize tokens using vocabulary
        indices = [self.word_to_idx.get(word, UNK_IDX) for word in tokens]

        # Truncate or pad sequence to max_len
        if len(indices) > self.max_len:
            indices = indices[: self.max_len]
        else:
            padding_length = self.max_len - len(indices)
            indices = indices + [PAD_IDX] * padding_length

        x_tensor = torch.tensor(indices, dtype=torch.long)
        y_tensor = torch.tensor(label, dtype=torch.float32)

        return x_tensor, y_tensor


def get_imdb_data_loaders(
    csv_path: str,
    max_vocab_size: int,
    max_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, int]:
    """Loads IMDb dataset, builds shared vocabulary, and creates train/val DataLoaders.

    Args:
        csv_path: File path to IMDb dataset CSV.
        max_vocab_size: Maximum vocabulary size.
        max_len: Maximum text sequence length.
        batch_size: DataLoader mini-batch size.

    Returns:
        Tuple[DataLoader, DataLoader, int]: Train DataLoader, Validation DataLoader,
            and total vocabulary size.
    """
    print(f"Loading IMDb dataset from {csv_path}...")
    df_imdb = pd.read_csv(csv_path)

    # Map sentiment labels to binary float target (positive=1.0, negative=0.0)
    df_imdb["sentiment"] = df_imdb["sentiment"].map({"positive": 1.0, "negative": 0.0})
    df_imdb = df_imdb.dropna(subset=["sentiment"])

    print("Cleaning IMDb reviews...")
    imdb_texts = [clean_text(text) for text in df_imdb["review"].tolist()]
    imdb_labels = df_imdb["sentiment"].tolist()

    # Load financial text tokens solely to include domain terms in shared vocabulary
    print("Reading Financial PhraseBank text to build joint vocabulary...")
    financial_texts = []
    with open(FINANCIAL_DATA, "r", encoding="latin-1") as f:
        for line in f:
            if "@" in line:
                text_part = line.split("@")[0]
                financial_texts.append(clean_text(text_part))

    print("Building joint vocabulary across datasets...")
    all_texts = imdb_texts + financial_texts
    word_to_idx = build_vocab(all_texts, max_vocab_size)
    vocab_size = len(word_to_idx)
    print(f"Final shared vocabulary size: {vocab_size}")

    print("Splitting IMDb data into Train (80%) and Validation (20%)...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        imdb_texts, imdb_labels, test_size=0.2, random_state=42
    )

    train_dataset = TextDataset(X_train, y_train, word_to_idx, max_len)
    valid_dataset = TextDataset(X_valid, y_valid, word_to_idx, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print("IMDb DataLoaders initialized successfully.")
    return train_loader, valid_loader, vocab_size


def get_financial_data_loaders(
    financial_path: str,
    imdb_path: str,
    max_vocab_size: int,
    max_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, int]:
    """Loads Financial PhraseBank dataset and creates train/val DataLoaders.

    Args:
        financial_path: Path to Financial PhraseBank sentences text file.
        imdb_path: Path to IMDb dataset CSV for vocabulary consistency.
        max_vocab_size: Maximum vocabulary size.
        max_len: Maximum text sequence length.
        batch_size: DataLoader mini-batch size.

    Returns:
        Tuple[DataLoader, DataLoader, int]: Train DataLoader, Validation DataLoader,
            and shared vocabulary size.
    """
    print("Rebuilding shared joint vocabulary...")

    df_imdb = pd.read_csv(imdb_path)
    imdb_texts = [clean_text(text) for text in df_imdb["review"].tolist()]

    fin_texts = []
    fin_labels = []

    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    with open(financial_path, "r", encoding="latin-1") as f:
        for line in f:
            if "@" in line:
                text_part, label_part = line.strip().split("@")
                fin_texts.append(clean_text(text_part))
                fin_labels.append(label_map[label_part.lower()])

    all_texts = imdb_texts + fin_texts
    word_to_idx = build_vocab(all_texts, max_vocab_size)

    print("Splitting Financial PhraseBank dataset into Train (80%) and Validation (20%)...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        fin_texts, fin_labels, test_size=0.2, random_state=42
    )

    train_dataset = TextDataset(X_train, y_train, word_to_idx, max_len)
    valid_dataset = TextDataset(X_valid, y_valid, word_to_idx, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Financial DataLoaders ready: {len(X_train)} train samples, {len(X_valid)} validation samples."
    )
    return train_loader, valid_loader, len(word_to_idx)
