"""Bước 3: huấn luyện + đánh giá trên tập test.

Ví dụ:
  python scripts/train.py --model bilstm
  python scripts/train.py --model all          # cnn, lstm, bilstm
  python scripts/train.py --model baselines    # TF-IDF + Logistic Regression / Linear SVM
  python scripts/train.py --model cnn --epochs 3 --batch-size 16   # chạy thử
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation.metrics import compute_metrics, probs_to_labels
from src.evaluation.plots import plot_confusion_matrix, plot_history
from src.models.networks import build_model
from src.models.trainer import fit, predict_proba
from src.utils.config import load_config, resolve, set_seed


def run_baselines(cfg, feat, rep):
    Xtr, Xte = sp.load_npz(feat / "X_train_tfidf.npz"), sp.load_npz(feat / "X_test_tfidf.npz")
    ytr, yte = np.load(feat / "y_train.npy"), np.load(feat / "y_test.npy")
    models = {
        "tfidf_logreg": LogisticRegression(**cfg["baselines"]["logistic_regression"]),
        "tfidf_linear_svm": LinearSVC(**cfg["baselines"]["linear_svm"]),
    }
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        res = compute_metrics(yte, clf.predict(Xte))
        (rep / f"metrics_{name}.json").write_text(json.dumps(res, indent=2))
        plot_confusion_matrix(res["confusion_matrix"], rep / "figures" / f"cm_{name}.png", name)
        print(name, {k: round(v, 4) for k, v in res.items() if k != "confusion_matrix"})


def run_deep(name, cfg, feat, art, rep, device):
    X = {n: np.load(feat / f"seq_{n}.npy") for n in ("train", "val", "test")}
    y = {n: np.load(feat / f"y_{n}.npy") for n in ("train", "val", "test")}
    emb = np.load(feat / "embedding_matrix.npy")
    model = build_model(name, emb.shape[0], emb.shape[1], emb, cfg)
    print(f"== {name.upper()} | params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    hist = fit(model, X["train"], y["train"], X["val"], y["val"], cfg["training"], device)
    probs = predict_proba(model, X["test"], device=device)
    res = compute_metrics(y["test"], probs_to_labels(probs))
    res["epochs_run"] = len(hist["loss"])

    (art / "models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), art / "models" / f"{name}.pt")
    (rep / f"metrics_{name}.json").write_text(json.dumps(res, indent=2))
    (rep / f"history_{name}.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, rep / "figures" / f"history_{name}.png", name.upper())
    plot_confusion_matrix(res["confusion_matrix"], rep / "figures" / f"cm_{name}.png", name.upper())
    print(name, {k: round(v, 4) for k, v in res.items() if k not in ("confusion_matrix", "epochs_run")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="all", choices=["cnn", "lstm", "bilstm", "all", "baselines"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    set_seed(cfg["seed"])

    feat, art, rep = resolve(cfg, "features_dir"), resolve(cfg, "artifacts_dir"), resolve(cfg, "reports_dir")
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if args.model == "baselines":
        run_baselines(cfg, feat, rep)
    else:
        for name in (["cnn", "lstm", "bilstm"] if args.model == "all" else [args.model]):
            run_deep(name, cfg, feat, art, rep, device)


if __name__ == "__main__":
    main()
