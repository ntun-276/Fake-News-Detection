"""Bước 2: xây dựng đặc trưng TF-IDF, FastText (mean pooling) và chuỗi chỉ số cho CNN/LSTM/BiLSTM.

Mọi bộ biểu diễn (TF-IDF, FastText, vocab) chỉ được fit trên tập TRAIN để tránh rò rỉ dữ liệu.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.fasttext import mean_pool, train_fasttext
from src.features.sequence import Vocab, embedding_matrix, pad_truncate
from src.features.tf_idf import build_tfidf, save_tfidf
from src.utils.config import load_config, resolve, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tok_col, label_col = cfg["dataset"]["tokenized_column"], cfg["dataset"]["label_column"]
    proc, feat, art = resolve(cfg, "processed_dir"), resolve(cfg, "features_dir"), resolve(cfg, "artifacts_dir") / "features"
    feat.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    parts = {n: pd.read_csv(proc / f"{n}.csv") for n in ("train", "val", "test")}
    texts = {n: d[tok_col].astype(str).tolist() for n, d in parts.items()}
    tokens = {n: [t.split() for t in v] for n, v in texts.items()}
    for n, d in parts.items():
        np.save(feat / f"y_{n}.npy", d[label_col].to_numpy())

    # ---- TF-IDF ----
    vec = build_tfidf(cfg["tfidf"])
    X_train = vec.fit_transform(texts["train"])
    save_tfidf(vec, X_train, art / "tfidf_vectorizer.joblib", feat / "X_train_tfidf.npz")
    for n in ("val", "test"):
        sp.save_npz(feat / f"X_{n}_tfidf.npz", vec.transform(texts[n]))
    print(f"TF-IDF vocab_size = {len(vec.vocabulary_)}")

    # ---- FastText ----
    ft = train_fasttext(tokens["train"], cfg["fasttext"], cfg["seed"])
    ft.save(str(art / "fasttext.model"))
    for n in ("train", "val", "test"):
        np.save(feat / f"X_{n}_fasttext.npy", mean_pool(ft, tokens[n]))

    # ---- Chuỗi chỉ số + ma trận embedding (cho CNN / LSTM / BiLSTM) ----
    sq = cfg["sequence"]
    vocab = Vocab.build(tokens["train"], sq["min_word_freq"], sq["pad_token"], sq["unk_token"])
    for n in ("train", "val", "test"):
        ids = [vocab.encode(t) for t in tokens[n]]
        np.save(feat / f"seq_{n}.npy", pad_truncate(ids, sq["max_seq_len"]))
    np.save(feat / "embedding_matrix.npy", embedding_matrix(vocab, ft))
    (art / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"vocab (sequence) = {len(vocab)} | max_seq_len = {sq['max_seq_len']}")


if __name__ == "__main__":
    main()
