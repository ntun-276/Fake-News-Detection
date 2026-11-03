"""Từ điển, chuyển token -> chỉ số, padding/truncating, ma trận embedding."""
from __future__ import annotations

from collections import Counter

import numpy as np


class Vocab:
    def __init__(self, pad: str = "<PAD>", unk: str = "<UNK>"):
        self.pad, self.unk = pad, unk
        self.itos: list[str] = [pad, unk]
        self.stoi: dict[str, int] = {pad: 0, unk: 1}

    @classmethod
    def build(cls, token_lists: list[list[str]], min_freq: int = 1, pad="<PAD>", unk="<UNK>") -> "Vocab":
        v = cls(pad, unk)
        counts = Counter(t for toks in token_lists for t in toks)
        for tok, c in counts.most_common():
            if c >= min_freq and tok not in v.stoi:
                v.stoi[tok] = len(v.itos)
                v.itos.append(tok)
        return v

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[self.unk]
        return [self.stoi.get(t, unk) for t in tokens]

    def to_dict(self) -> dict:
        return {"itos": self.itos, "pad": self.pad, "unk": self.unk}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        v = cls(d["pad"], d["unk"])
        v.itos = list(d["itos"])
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


def pad_truncate(seqs: list[list[int]], max_len: int, pad_id: int = 0) -> np.ndarray:
    """Padding thêm giá trị pad vào cuối chuỗi ngắn; truncating cắt phần cuối chuỗi dài."""
    out = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        out[i, : len(s)] = s
    return out


def embedding_matrix(vocab: Vocab, ft_model) -> np.ndarray:
    dim = ft_model.vector_size
    mat = np.zeros((len(vocab), dim), dtype=np.float32)
    for tok, idx in vocab.stoi.items():
        if idx >= 2:  # bỏ PAD, UNK
            mat[idx] = ft_model.wv[tok]
    return mat
