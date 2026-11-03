"""Đồ thị cho báo cáo: phân bố nhãn/độ dài, loss-accuracy, confusion matrix."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_label_distribution(labels, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=labels, ax=ax, color="#3274a1")
    ax.set(title="Label Distribution", xlabel="Label", ylabel="Number of Samples")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_length_distribution(lengths, path: Path, xlabel="Number of Words"):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(lengths, bins=40, ax=ax)
    ax.set(title="Text Length Distribution", xlabel=xlabel, ylabel="Frequency")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(history: dict, path: Path, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set(title=f"{title} {key}".strip(), xlabel="epoch", ylabel=key)
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion_matrix(cm, path: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
