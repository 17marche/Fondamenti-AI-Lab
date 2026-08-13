"""Main script for IMDb general sentiment pre-training stage."""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE,
    BIDIRECTIONAL,
    DEVICE,
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    IMDB_CSV,
    LEARNING_RATE_PRETRAIN,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    MODEL_SAVE_PATH,
    N_LAYERS,
    PAD_IDX,
    PRETRAIN_EPOCHS,
)
from data_prep import get_imdb_data_loaders
from engine import evaluate, train_epoch
from model import SentimentLSTM


def main() -> None:
    """Runs the IMDb pre-training pipeline and saves the best model checkpoint."""
    print(f"--- Starting IMDb Pre-Training Pipeline (Device: {DEVICE}) ---")

    # Load IMDb dataset and shared vocabulary
    train_loader, valid_loader, vocab_size = get_imdb_data_loaders(
        csv_path=IMDB_CSV,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_len=MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
    )

    # Initialize SentimentLSTM model for binary classification
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    ).to(DEVICE)

    # Optimizer and loss function setup
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_PRETRAIN, weight_decay=1e-2
    )
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    best_valid_loss = float("inf")

    # Training loop across epochs
    for epoch in range(PRETRAIN_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, is_binary=True
        )
        valid_loss, valid_acc = evaluate(
            model, valid_loader, criterion, DEVICE, is_binary=True
        )

        print(f"Epoch {epoch + 1:02d}/{PRETRAIN_EPOCHS:02d}")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%"
        )

        # Checkpoint model with lowest validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"\t => Best model checkpoint saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()