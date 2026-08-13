"""Main script for Financial PhraseBank sentiment fine-tuning stage."""

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
    FINANCIAL_CLASSES,
    FINANCIAL_DATA,
    FINETUNE_EPOCHS,
    HIDDEN_DIM,
    IMDB_CSV,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    MODEL_SAVE_PATH,
    N_LAYERS,
    PAD_IDX,
    WARMUP_EPOCHS,
)
from data_prep import get_financial_data_loaders
from engine import evaluate, train_epoch
from model import SentimentLSTM


def main() -> None:
    """Runs two-stage fine-tuning pipeline on Financial PhraseBank dataset."""
    print(f"--- Starting Financial PhraseBank Fine-Tuning Pipeline ({DEVICE}) ---")

    # Load financial dataset and rebuild matching joint vocabulary
    train_loader, valid_loader, vocab_size = get_financial_data_loaders(
        financial_path=FINANCIAL_DATA,
        imdb_path=IMDB_CSV,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_len=MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
    )

    # Instantiate model structure matching pre-trained state
    print("\nInstantiating SentimentLSTM model and loading IMDb pre-trained weights...")
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

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    print("IMDb pre-trained weights loaded successfully.")

    # Adapt classification head from 1 binary logit to 3 financial sentiment classes
    print("\nReplacing final classification layer for 3-class financial sentiment...")
    model.classifier[3] = nn.Linear(
        in_features=HIDDEN_DIM, out_features=FINANCIAL_CLASSES
    )
    model = model.to(DEVICE)

    # Class weighting via Square Root Balancing to address neutral class prevalence
    # Classes: 0 (Negative), 1 (Neutral), 2 (Positive)
    weights = torch.tensor([2.8, 1.0, 1.8], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights).to(DEVICE)

    # Evaluate zero-shot / baseline performance before fine-tuning
    print("\n--- Baseline Evaluation (Untrained MLP Head) ---")
    baseline_loss, baseline_acc = evaluate(
        model, valid_loader, criterion, DEVICE, is_binary=False
    )
    print(f"\t-> Baseline Validation Loss:     {baseline_loss:.3f}")
    print(f"\t-> Baseline Validation Accuracy: {baseline_acc * 100:.2f}%")
    print("-" * 65)

    # Stage 1: Freeze Encoder & Warm-Up MLP Classifier
    print("\n--- Stage 1: Classifier Warm-Up (Encoder Frozen) ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Active trainable parameters: {trainable_params:,} (MLP head only)")

    optimizer_warmup = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-3,
        weight_decay=1e-2,
    )

    best_warmup_loss = float("inf")
    warmup_model_path = MODEL_SAVE_PATH.replace(".pth", "_finetuned.pth")

    for epoch in range(WARMUP_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer_warmup, criterion, DEVICE, is_binary=False
        )
        valid_loss, valid_acc = evaluate(
            model, valid_loader, criterion, DEVICE, is_binary=False
        )

        print(
            f"Warm-Up Epoch {epoch + 1:02d}/{WARMUP_EPOCHS:02d} | "
            f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {valid_acc * 100:.2f}%"
        )

        if valid_loss < best_warmup_loss:
            best_warmup_loss = valid_loss
            torch.save(model.state_dict(), warmup_model_path)

    # Stage 1 Benchmark
    print("\nPost Warm-Up Evaluation:")
    model.load_state_dict(torch.load(warmup_model_path, map_location=DEVICE))
    _, warmup_acc = evaluate(
        model, valid_loader, criterion, DEVICE, is_binary=False
    )
    print(f"\t-> Warm-Up Validation Loss:     {best_warmup_loss:.3f}")
    print(f"\t-> Warm-Up Validation Accuracy: {warmup_acc * 100:.2f}%")
    print("-" * 65)

    # Stage 2: Full Fine-Tuning with Differential Learning Rates
    print("\n--- Stage 2: Selective Full Fine-Tuning (Bi-LSTM + MLP Unfrozen) ---")
    for param in model.lstm.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params_full = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Active trainable parameters: {trainable_params_full:,} (Bi-LSTM + MLP)"
    )

    # Differential learning rates: 1e-4 for LSTM encoder, 1e-3 for MLP classifier
    optimizer_full = optim.AdamW(
        [
            {
                "params": filter(
                    lambda p: p.requires_grad, model.lstm.parameters()
                ),
                "lr": 1e-4,
                "weight_decay": 1e-4,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad, model.classifier.parameters()
                ),
                "lr": 1e-3,
                "weight_decay": 1e-2,
            },
        ]
    )

    best_final_loss = float("inf")
    patience = 7
    patience_counter = 0
    final_model_path = MODEL_SAVE_PATH.replace(".pth", "_final_best.pth")

    for epoch in range(FINETUNE_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer_full, criterion, DEVICE, is_binary=False
        )
        valid_loss, valid_acc = evaluate(
            model, valid_loader, criterion, DEVICE, is_binary=False
        )

        print(
            f"Fine-Tuning Epoch {epoch + 1:02d}/{FINETUNE_EPOCHS:02d} | "
            f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {valid_acc * 100:.2f}%"
        )

        if valid_loss < best_final_loss:
            best_final_loss = valid_loss
            torch.save(model.state_dict(), final_model_path)
            patience_counter = 0
            print(f"\t => Saved best model checkpoint (Val Loss: {valid_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n[Early Stopping] Triggered at epoch {epoch + 1} due to non-improving validation loss."
                )
                break

    # Stage 2 Final Benchmark
    print("\n--- Final Model Evaluation ---")
    model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
    _, final_acc = evaluate(
        model, valid_loader, criterion, DEVICE, is_binary=False
    )
    print(f"\t-> Final Validation Loss:     {best_final_loss:.3f}")
    print(f"\t-> Final Validation Accuracy: {final_acc * 100:.2f}%")
    print("-" * 65)


if __name__ == "__main__":
    main()