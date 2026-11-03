"""Biểu diễn TF-IDF (unigram + bigram) trên văn bản đã tách từ."""
from __future__ import annotations

from pathlib import Path

import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(cfg: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=str.split,       # văn bản đã được tách từ ở bước tiền xử lý
        token_pattern=None,
        lowercase=False,
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
    )


def save_tfidf(vectorizer: TfidfVectorizer, X, vec_path: Path, matrix_path: Path) -> None:
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vec_path)
    sp.save_npz(matrix_path, X)
