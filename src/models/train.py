from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import clean_dataset
from src.features.build_features import build_vocabulary, pad_or_truncate_sequences, texts_to_sequences
from src.models.evaluate import compute_classification_metrics
from src.models.lstm_baseline import LSTMBaselineClassifier


@dataclass
class SequenceBundle:
    train_sequences: list[list[int]]
    val_sequences: Optional[list[list[int]]]
    test_sequences: Optional[list[list[int]]]
    vocab_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _coerce_sequence(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list-like input_ids, got: {value}")
        return [int(item) for item in parsed]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    raise ValueError(f"Unsupported sequence value: {value!r}")


def _sequences_from_column(df: pd.DataFrame, sequence_col: str, max_length: int) -> list[list[int]]:
    if sequence_col not in df.columns:
        raise ValueError(f"Missing sequence column '{sequence_col}'")
    parsed = [_coerce_sequence(item) for item in df[sequence_col].tolist()]
    return pad_or_truncate_sequences(parsed, max_length=max_length, padding="post", truncating="post", pad_value=0)


def _sequences_from_tokenized(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    tokenized_col: str,
    max_length: int,
    min_freq: int,
    max_vocab_size: Optional[int],
) -> SequenceBundle:
    if tokenized_col not in train_df.columns:
        raise ValueError(f"Missing tokenized column '{tokenized_col}' in train split")

    train_tokenized = train_df[tokenized_col].fillna("").astype(str).tolist()
    vocab = build_vocabulary(train_tokenized, min_freq=min_freq, max_vocab_size=max_vocab_size)

    train_sequences = pad_or_truncate_sequences(
        texts_to_sequences(train_tokenized, vocab),
        max_length=max_length,
        padding="post",
        truncating="post",
        pad_value=0,
    )

    val_sequences: Optional[list[list[int]]] = None
    if val_df is not None:
        if tokenized_col not in val_df.columns:
            raise ValueError(f"Missing tokenized column '{tokenized_col}' in val split")
        val_sequences = pad_or_truncate_sequences(
            texts_to_sequences(val_df[tokenized_col].fillna("").astype(str).tolist(), vocab),
            max_length=max_length,
            padding="post",
            truncating="post",
            pad_value=0,
        )

    test_sequences: Optional[list[list[int]]] = None
    if test_df is not None:
        if tokenized_col not in test_df.columns:
            raise ValueError(f"Missing tokenized column '{tokenized_col}' in test split")
        test_sequences = pad_or_truncate_sequences(
            texts_to_sequences(test_df[tokenized_col].fillna("").astype(str).tolist(), vocab),
            max_length=max_length,
            padding="post",
            truncating="post",
            pad_value=0,
        )

    return SequenceBundle(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        vocab_size=len(vocab),
    )


def _sequences_with_clean_pipeline(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    text_col: str,
    max_length: int,
    min_freq: int,
    max_vocab_size: Optional[int],
    abbreviation_csv: Optional[Path],
) -> SequenceBundle:
    if text_col not in train_df.columns:
        raise ValueError(f"Missing text column '{text_col}' in train split")

    train_clean = clean_dataset(
        train_df,
        text_col=text_col,
        max_length=max_length,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        abbreviation_csv=abbreviation_csv,
    )
    train_sequences = train_clean["input_ids"].tolist()

    vocab = build_vocabulary(
        train_clean["tokenized_text"].fillna("").astype(str).tolist(),
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )

    val_sequences: Optional[list[list[int]]] = None
    if val_df is not None:
        if text_col not in val_df.columns:
            raise ValueError(f"Missing text column '{text_col}' in val split")
        val_clean = clean_dataset(
            val_df,
            text_col=text_col,
            max_length=max_length,
            vocab=vocab,
            abbreviation_csv=abbreviation_csv,
        )
        val_sequences = val_clean["input_ids"].tolist()

    test_sequences: Optional[list[list[int]]] = None
    if test_df is not None:
        if text_col not in test_df.columns:
            raise ValueError(f"Missing text column '{text_col}' in test split")
        test_clean = clean_dataset(
            test_df,
            text_col=text_col,
            max_length=max_length,
            vocab=vocab,
            abbreviation_csv=abbreviation_csv,
        )
        test_sequences = test_clean["input_ids"].tolist()

    return SequenceBundle(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        vocab_size=len(vocab),
    )


def _encode_labels(labels: Iterable[object]) -> tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    normalized = pd.Series(labels).astype(str)
    classes = sorted(normalized.unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded = normalized.map(label_to_id).astype(int).to_numpy()
    return encoded, label_to_id, id_to_label


def _tensor_dataset(sequences: list[list[int]], labels: np.ndarray) -> TensorDataset:
    x_tensor = torch.tensor(sequences, dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def _evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    return avg_loss, np.array(y_true), np.array(y_pred)


def train_lstm_baseline(
    train_path: Path,
    output_dir: Path,
    val_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    label_col: str = "Label",
    text_col: str = "Maintext",
    tokenized_col: str = "tokenized_text",
    sequence_col: str = "input_ids",
    max_length: int = 64,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    abbreviation_csv: Optional[Path] = Path("data/vietnamese_abbreviation_normalization.csv"),
    embedding_dim: int = 128,
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
    bidirectional: bool = False,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    seed: int = 42,
    force_rebuild_sequences: bool = False,
    require_input_ids: bool = True,
    model_filename: str = "lstm_baseline.pt",
    architecture_name: str = "lstm_baseline",
    phase_name: str = "phase1",
) -> dict:
    set_seed(seed)

    train_df = _read_csv(train_path)
    val_df = _read_csv(val_path) if val_path else None
    test_df = _read_csv(test_path) if test_path else None

    if label_col not in train_df.columns:
        raise ValueError(f"Missing label column '{label_col}' in train split")

    has_input_ids = (
        sequence_col in train_df.columns
        and (val_df is None or sequence_col in val_df.columns)
        and (test_df is None or sequence_col in test_df.columns)
    )
    if require_input_ids and not has_input_ids:
        raise ValueError(
            "Missing required input_ids column across provided splits. "
            "Please run preprocessing first and train with files that already contain 'input_ids'."
        )

    if force_rebuild_sequences:
        raise ValueError("force_rebuild_sequences is disabled in input_ids-only mode")

    train_sequences = _sequences_from_column(train_df, sequence_col, max_length)
    val_sequences = _sequences_from_column(val_df, sequence_col, max_length) if val_df is not None else None
    test_sequences = _sequences_from_column(test_df, sequence_col, max_length) if test_df is not None else None

    sequence_bundle = SequenceBundle(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        vocab_size=max(max((max(seq) for seq in train_sequences), default=0) + 1, 2),
    )
    sequence_mode = "input_ids"

    y_train, label_to_id, id_to_label = _encode_labels(train_df[label_col])
    num_classes = len(label_to_id)
    if num_classes < 2:
        raise ValueError("Need at least 2 label classes for classification")

    train_dataset = _tensor_dataset(sequence_bundle.train_sequences, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    y_val: Optional[np.ndarray] = None
    val_loader: Optional[DataLoader] = None
    if val_df is not None and sequence_bundle.val_sequences is not None:
        if label_col not in val_df.columns:
            raise ValueError(f"Missing label column '{label_col}' in val split")
        y_val = pd.Series(val_df[label_col]).astype(str).map(label_to_id).fillna(-1).astype(int).to_numpy()
        if (y_val < 0).any():
            raise ValueError("Validation split contains unknown labels not seen in train split")
        val_dataset = _tensor_dataset(sequence_bundle.val_sequences, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMBaselineClassifier(
        vocab_size=sequence_bundle.vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict] = []
    best_val_f1 = -1.0
    best_state_dict = model.state_dict()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)

        train_loss = total_train_loss / max(len(train_loader.dataset), 1)
        epoch_log: dict = {"epoch": epoch, "train_loss": round(train_loss, 6)}

        if val_loader is not None:
            val_loss, y_val_true, y_val_pred = _evaluate_model(model, val_loader, criterion, device)
            metrics = compute_classification_metrics(y_val_true, y_val_pred)
            epoch_log.update(
                {
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(metrics["accuracy"], 6),
                    "val_precision": round(metrics["precision"], 6),
                    "val_recall": round(metrics["recall"], 6),
                    "val_f1": round(metrics["f1"], 6),
                }
            )

            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append(epoch_log)

    if val_loader is not None and best_val_f1 >= 0:
        model.load_state_dict(best_state_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_filename

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "architecture_name": architecture_name,
            "phase_name": phase_name,
            "vocab_size": sequence_bundle.vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "num_classes": num_classes,
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "sequence_mode": sequence_mode,
        },
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }
    torch.save(checkpoint, model_path)

    final_result: dict = {
        "device": str(device),
        "model_path": str(model_path),
        "history": history,
        "label_to_id": label_to_id,
        "sequence_mode": sequence_mode,
    }

    if test_df is not None and sequence_bundle.test_sequences is not None:
        if label_col not in test_df.columns:
            raise ValueError(f"Missing label column '{label_col}' in test split")

        y_test = pd.Series(test_df[label_col]).astype(str).map(label_to_id).fillna(-1).astype(int).to_numpy()
        if (y_test < 0).any():
            raise ValueError("Test split contains unknown labels not seen in train split")

        test_dataset = _tensor_dataset(sequence_bundle.test_sequences, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        _, y_true, y_pred = _evaluate_model(model, test_loader, criterion, device)
        final_result["test_metrics"] = compute_classification_metrics(y_true, y_pred)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")

    return final_result


def train_bilstm_phase2(
    train_path: Path,
    output_dir: Path,
    val_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    label_col: str = "Label",
    text_col: str = "Maintext",
    tokenized_col: str = "tokenized_text",
    sequence_col: str = "input_ids",
    max_length: int = 64,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    abbreviation_csv: Optional[Path] = Path("data/vietnamese_abbreviation_normalization.csv"),
    embedding_dim: int = 128,
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    seed: int = 42,
    force_rebuild_sequences: bool = False,
    require_input_ids: bool = True,
) -> dict:
    """Train Phase-2 BiLSTM with the same setup as phase-1 baseline."""

    return train_lstm_baseline(
        train_path=train_path,
        output_dir=output_dir,
        val_path=val_path,
        test_path=test_path,
        label_col=label_col,
        text_col=text_col,
        tokenized_col=tokenized_col,
        sequence_col=sequence_col,
        max_length=max_length,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        abbreviation_csv=abbreviation_csv,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        force_rebuild_sequences=force_rebuild_sequences,
        require_input_ids=require_input_ids,
        model_filename="bilstm_phase2.pt",
        architecture_name="bilstm_phase2",
        phase_name="phase2",
    )



