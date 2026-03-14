from __future__ import annotations

from typing import Iterable, List

import numpy as np

try:
    from gensim.models import FastText
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "gensim is required for FastText features. Install dependencies from requirements.txt"
    ) from exc


def _to_token_lists(tokenized_texts: Iterable[str]) -> List[List[str]]:
    token_lists: List[List[str]] = []
    for text in tokenized_texts:
        if text is None:
            token_lists.append([])
        elif isinstance(text, str):
            token_lists.append(text.split())
        else:
            token_lists.append(str(text).split())
    return token_lists


def train_fasttext_model(
    tokenized_texts: Iterable[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 10,
    sg: int = 1,
    workers: int = 1,
    seed: int = 42,
) -> FastText:
    sentences = _to_token_lists(tokenized_texts)
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        seed=seed,
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model


def documents_to_fasttext_vectors(
    tokenized_texts: Iterable[str],
    model: FastText,
    pooling: str = "mean",
) -> np.ndarray:
    if pooling not in {"mean", "sum"}:
        raise ValueError("pooling must be 'mean' or 'sum'")

    sentences = _to_token_lists(tokenized_texts)
    dim = model.wv.vector_size
    result = np.zeros((len(sentences), dim), dtype=np.float32)

    for idx, tokens in enumerate(sentences):
        if not tokens:
            continue

        vectors = np.vstack([model.wv[token] for token in tokens])
        result[idx] = vectors.mean(axis=0) if pooling == "mean" else vectors.sum(axis=0)

    return result

