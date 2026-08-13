"""Training and evaluation step routines for PyTorch neural network execution."""

from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_accuracy(
    preds: torch.Tensor, y: torch.Tensor, is_binary: bool = True
) -> torch.Tensor:
    """Calculates accuracy metric for binary or multiclass predictions.

    Args:
        preds: Raw model output logits.
        y: Ground truth target labels.
        is_binary: If True, treats task as binary classification; otherwise, multiclass.

    Returns:
        torch.Tensor: Scalar accuracy value as a floating point tensor.
    """
    if is_binary:
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
    else:
        probabilities = torch.softmax(preds, dim=1)
        predicted_classes = probabilities.argmax(dim=1)
        correct = (predicted_classes == y).float()

    acc = correct.sum() / len(correct)
    return acc


def train_epoch(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool = True,
) -> Tuple[float, float]:
    """Executes a single training epoch over the dataset iterator.

    Args:
        model: PyTorch SentimentLSTM model instance.
        iterator: DataLoader yielding batch tuples of (text, labels).
        optimizer: PyTorch optimizer instance.
        criterion: Loss function module.
        device: Device execution target (CUDA or CPU).
        is_binary: Flag indicating binary classification mode.

    Returns:
        Tuple[float, float]: Mean loss and mean accuracy over the epoch.
    """
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()

    for i, batch in enumerate(iterator):
        text, labels = batch
        text = text.to(device)

        if is_binary:
            labels = labels.float().to(device)
        else:
            labels = labels.long().to(device)

        optimizer.zero_grad()

        predictions = model(text)

        if is_binary:
            predictions = predictions.squeeze(1)

        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels, is_binary)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if (i + 1) % 50 == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{current_time}] Batch {i + 1}/{len(iterator)} | "
                f"Loss: {loss.item():.4f} | Acc: {acc.item():.4f}"
            )

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(
    model: nn.Module,
    iterator: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool = True,
) -> Tuple[float, float]:
    """Evaluates the model on validation or test dataset iterator.

    Args:
        model: PyTorch SentimentLSTM model instance.
        iterator: DataLoader yielding batch tuples of (text, labels).
        criterion: Loss function module.
        device: Device execution target (CUDA or CPU).
        is_binary: Flag indicating binary classification mode.

    Returns:
        Tuple[float, float]: Mean validation loss and accuracy.
    """
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text = text.to(device)

            if is_binary:
                labels = labels.float().to(device)
            else:
                labels = labels.long().to(device)

            predictions = model(text)

            if is_binary:
                predictions = predictions.squeeze(1)

            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels, is_binary)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
