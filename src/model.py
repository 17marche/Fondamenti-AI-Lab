"""Bidirectional LSTM neural network module for sentiment classification."""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """Bidirectional LSTM architecture with Masked Mean Pooling and an MLP classifier.

    Attributes:
        hidden_dim (int): Number of features in the LSTM hidden state.
        n_layers (int): Number of recurrent layers.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
        pad_idx (int): Vocabulary index used for padding sequences.
        embedding (nn.Embedding): Token embedding layer.
        lstm (nn.LSTM): Recurrent network encoder.
        classifier (nn.Sequential): Multilayer Perceptron classification head.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        bidirectional: bool,
        dropout: float,
        pad_idx: int,
    ):
        """Initializes the SentimentLSTM model components.

        Args:
            vocab_size: Total size of the vocabulary.
            embedding_dim: Dimension of token embeddings.
            hidden_dim: Number of features in the hidden state of the LSTM.
            output_dim: Number of output classes (1 for binary, N for multiclass).
            n_layers: Number of stacked LSTM layers.
            bidirectional: Whether to process sequences bidirectionally.
            dropout: Dropout probability applied between layers and in the MLP.
            pad_idx: Token index representing padding tokens.
        """
        super(SentimentLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

        # Token embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        # Recurrent feature extractor (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # Multilayer Perceptron (MLP) classification head
        mlp_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_features=mlp_input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Executes forward pass with Masked Mean Pooling over non-padding tokens.

        Args:
            text: Input tensor of shape `(batch_size, sequence_length)` containing
                token indices.

        Returns:
            torch.Tensor: Unnormalized output logits of shape `(batch_size, output_dim)`.
        """
        # Create mask of shape (batch_size, sequence_length, 1) ignoring padding tokens
        mask = (text != self.pad_idx).unsqueeze(-1).float()

        # Map token indices to embedding vectors: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)

        # Compute LSTM outputs: (batch_size, seq_len, hidden_dim * num_directions)
        output, _ = self.lstm(embedded)

        # Mask padding states to exclude them from pooling
        masked_output = output * mask

        # Perform masked mean pooling across the sequence dimension
        sum_masked = torch.sum(masked_output, dim=1)
        num_tokens = torch.sum(mask, dim=1).clamp(min=1e-9)
        pooled_output = sum_masked / num_tokens

        # Pass through classification head
        logits = self.classifier(pooled_output)

        return logits
