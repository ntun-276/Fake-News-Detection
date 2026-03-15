from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import train_lstm_baseline


def run_smoke_test() -> None:
    output_dir = PROJECT_ROOT / "artifacts" / "lstm_phase1_smoke"
    smoke_data_dir = PROJECT_ROOT / "artifacts" / "tmp_lstm_smoke_data"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if smoke_data_dir.exists():
        shutil.rmtree(smoke_data_dir)
    smoke_data_dir.mkdir(parents=True, exist_ok=True)

    train_path = smoke_data_dir / "train.csv"
    val_path = smoke_data_dir / "val.csv"

    train_df = pd.DataFrame(
        {
            "Label": ["fake", "real", "fake", "real", "fake", "real"],
            "input_ids": [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [1, 7, 3],
                [9, 10, 11],
                [1, 2],
                [8, 9, 10, 11],
            ],
        }
    )
    val_df = pd.DataFrame(
        {
            "Label": ["fake", "real"],
            "input_ids": [[1, 2, 3], [8, 9, 10]],
        }
    )
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    result = train_lstm_baseline(
        train_path=train_path,
        val_path=val_path,
        output_dir=output_dir,
        epochs=1,
        batch_size=4,
        hidden_size=64,
        embedding_dim=64,
        max_length=32,
        seed=123,
        require_input_ids=True,
    )

    model_path = Path(result["model_path"])
    metrics_path = output_dir / "metrics.json"

    assert model_path.exists(), "Model checkpoint was not created"
    assert metrics_path.exists(), "Metrics file was not created"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "history" in metrics and metrics["history"], "Training history is missing"
    assert result["sequence_mode"] == "input_ids", "LSTM is not running in input_ids mode"

    print("LSTM smoke test passed")


if __name__ == "__main__":
    run_smoke_test()


