"""Kiến trúc CNN / LSTM / BiLSTM cho phân loại văn bản (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Base(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, emb_matrix: np.ndarray | None, freeze: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = not freeze


class TextCNN(_Base):
    """Kim (2014): nhiều kernel size + max-over-time pooling."""

    def __init__(self, vocab_size, emb_dim, emb_matrix=None, freeze=False,
                 kernel_sizes=(3, 4, 5), num_filters=100, dropout=0.5):
        super().__init__(vocab_size, emb_dim, emb_matrix, freeze)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        e = self.embedding(x).transpose(1, 2)                 # (B, E, T)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1))).squeeze(1)


class _RNNClassifier(_Base):
    bidirectional = False

    def __init__(self, vocab_size, emb_dim, emb_matrix=None, freeze=False,
                 hidden_size=128, num_layers=1, dropout=0.5):
        super().__init__(vocab_size, emb_dim, emb_matrix, freeze)
        self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, batch_first=True,
                           bidirectional=self.bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        mask = (x != 0)
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(h)).squeeze(1)


class LSTMClassifier(_RNNClassifier):
    bidirectional = False


class BiLSTMClassifier(_RNNClassifier):
    bidirectional = True


def build_model(name: str, vocab_size: int, emb_dim: int, emb_matrix, cfg: dict) -> nn.Module:
    drop = cfg["training"]["dropout"]
    freeze = cfg["models"]["freeze_embedding"]
    if name == "cnn":
        m = cfg["models"]["cnn"]
        return TextCNN(vocab_size, emb_dim, emb_matrix, freeze, tuple(m["kernel_sizes"]), m["num_filters"], drop)
    if name in ("lstm", "bilstm"):
        m = cfg["models"][name]
        cls = LSTMClassifier if name == "lstm" else BiLSTMClassifier
        return cls(vocab_size, emb_dim, emb_matrix, freeze, m["hidden_size"], m["num_layers"], drop)
    raise ValueError(f"Model không hợp lệ: {name}")
