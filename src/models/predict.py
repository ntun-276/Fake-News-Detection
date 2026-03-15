from __future__ import annotations

from pathlib import Path

import torch

from src.models.lstm_baseline import LSTMBaselineClassifier


def load_lstm_checkpoint(model_path: Path) -> tuple[LSTMBaselineClassifier, dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]

    model = LSTMBaselineClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        num_classes=config["num_classes"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_from_sequences(model: LSTMBaselineClassifier, sequences: list[list[int]]) -> list[int]:
    if not sequences:
        return []

    with torch.no_grad():
        x_tensor = torch.tensor(sequences, dtype=torch.long)
        logits = model(x_tensor)
        predictions = torch.argmax(logits, dim=1)
    return predictions.cpu().numpy().tolist()

