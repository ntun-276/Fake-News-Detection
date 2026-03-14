from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse

try:
    from gensim.models import FastText
except ImportError:
    FastText = Any  # type: ignore[assignment]


def save_tfidf_vectorizer(vectorizer: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    return path


def load_tfidf_vectorizer(path: str | Path) -> Any:
    return joblib.load(path)


def save_sparse_matrix(matrix: sparse.spmatrix, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, matrix)
    return path


def load_sparse_matrix(path: str | Path) -> sparse.spmatrix:
    return sparse.load_npz(path)


def save_fasttext_model(model: FastText, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def load_fasttext_model(path: str | Path) -> FastText:
    return FastText.load(str(path))


def save_dense_array(array: np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path


def load_dense_array(path: str | Path) -> np.ndarray:
    return np.load(path)

