from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import train_bilstm_phase2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train phase-2 BiLSTM with the same setup as phase-1 baseline."
    )
    parser.add_argument("--train-path", default="data/update_data/update_train_data.csv")
    parser.add_argument("--val-path", default="data/update_data/update_val_data.csv")
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default="artifacts/bilstm_phase2")

    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--text-col", default="Maintext")
    parser.add_argument("--tokenized-col", default="tokenized_text")
    parser.add_argument("--sequence-col", default="input_ids")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--abbreviation-csv", default="data/vietnamese_abbreviation_normalization.csv")

    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    result = train_bilstm_phase2(
        train_path=Path(args.train_path),
        val_path=Path(args.val_path) if args.val_path else None,
        test_path=Path(args.test_path) if args.test_path else None,
        output_dir=Path(args.output_dir),
        label_col=args.label_col,
        text_col=args.text_col,
        tokenized_col=args.tokenized_col,
        sequence_col=args.sequence_col,
        max_length=args.max_length,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        abbreviation_csv=Path(args.abbreviation_csv) if args.abbreviation_csv else None,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        require_input_ids=True,
    )

    if result["sequence_mode"] != "input_ids":
        raise RuntimeError("BiLSTM must run with input_ids mode")

    print("Training completed.")
    print(f"Model saved: {result['model_path']}")
    print(f"Sequence mode used: {result['sequence_mode']}")

    if result["history"]:
        print("Last epoch log:")
        print(json.dumps(result["history"][-1], ensure_ascii=False, indent=2))

    if "test_metrics" in result:
        print("Test metrics:")
        print(json.dumps(result["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


