"""Word embedding FastText (gensim) + mean pooling cho mỗi văn bản."""
from __future__ import annotations

import numpy as np
from gensim.models import FastText


def train_fasttext(token_lists: list[list[str]], cfg: dict, seed: int = 42) -> FastText:
    return FastText(
        sentences=token_lists,
        vector_size=cfg["vector_size"],
        window=cfg["window"],
        min_count=cfg["min_count"],
        epochs=cfg["epochs"],
        workers=cfg.get("workers", 2),
        seed=seed,
    )


def mean_pool(model: FastText, token_lists: list[list[str]]) -> np.ndarray:
    dim = model.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        if toks:
            out[i] = np.mean([model.wv[t] for t in toks], axis=0)
    return out
