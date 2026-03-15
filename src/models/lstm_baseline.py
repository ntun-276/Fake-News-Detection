from __future__ import annotations

import torch
from torch import nn


class LSTMBaselineClassifier(nn.Module):
    """Simple LSTM text classifier for phase-1 baseline experiments."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        _, (hidden_states, _) = self.lstm(embeddings)

        if self.lstm.bidirectional:
            # Concatenate final forward and backward hidden states from the top LSTM layer.
            last_hidden = torch.cat((hidden_states[-2], hidden_states[-1]), dim=1)
        else:
            last_hidden = hidden_states[-1]

        logits = self.classifier(self.dropout(last_hidden))
        return logits

