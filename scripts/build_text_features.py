import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.artifact_io import (
    save_dense_array,
    save_fasttext_model,
    save_sparse_matrix,
    save_tfidf_vectorizer,
)
from src.features.fasttext_features import documents_to_fasttext_vectors, train_fasttext_model
from src.features.tfidf_features import build_tfidf_vectorizer, transform_tfidf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build text representations from tokenized_text column.")
    parser.add_argument("--train-path", required=True, help="Preprocessed train CSV path")
    parser.add_argument("--val-path", default=None, help="Optional preprocessed validation CSV path")
    parser.add_argument("--test-path", default=None, help="Optional preprocessed test CSV path")
    parser.add_argument(
        "--tokenized-col",
        default="tokenized_text",
        help="Must be tokenized_text (kept for compatibility)",
    )
    parser.add_argument(
        "--mode",
        choices=["tfidf", "fasttext", "both"],
        default="both",
        help="Feature mode to build",
    )
    parser.add_argument("--artifact-dir", default="artifacts/features", help="Directory to save fitted models")
    parser.add_argument("--output-dir", default="data/features", help="Directory to save feature matrices")

    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--tfidf-min-df", type=float, default=2)
    parser.add_argument("--tfidf-max-df", type=float, default=0.95)
    parser.add_argument("--tfidf-max-features", type=int, default=None)

    parser.add_argument("--ft-vector-size", type=int, default=100)
    parser.add_argument("--ft-window", type=int, default=5)
    parser.add_argument("--ft-min-count", type=int, default=2)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--ft-sg", type=int, default=1)
    parser.add_argument("--ft-workers", type=int, default=1)
    parser.add_argument("--ft-seed", type=int, default=42)
    parser.add_argument("--ft-pooling", choices=["mean", "sum"], default="mean")
    return parser


def _read_split(path: str | None, tokenized_col: str) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path)
    if tokenized_col not in df.columns:
        raise ValueError(f"Missing required column '{tokenized_col}' in {path}")
    return df


def _as_text_series(df: pd.DataFrame, tokenized_col: str) -> pd.Series:
    # Normalize whitespace once so both TF-IDF and FastText read identical tokenized input.
    return df[tokenized_col].fillna("").astype(str).map(lambda text: " ".join(text.split()))


def _save_metadata(metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()

    if args.tokenized_col != "tokenized_text":
        raise ValueError(
            "This pipeline only supports the preprocessed column 'tokenized_text'. "
            "Please run with --tokenized-col tokenized_text"
        )

    train_df = _read_split(args.train_path, args.tokenized_col)
    val_df = _read_split(args.val_path, args.tokenized_col)
    test_df = _read_split(args.test_path, args.tokenized_col)

    assert train_df is not None
    train_texts = _as_text_series(train_df, args.tokenized_col)
    val_texts = _as_text_series(val_df, args.tokenized_col) if val_df is not None else None
    test_texts = _as_text_series(test_df, args.tokenized_col) if test_df is not None else None

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    metadata: dict = {
        "tokenized_col": args.tokenized_col,
        "mode": args.mode,
        "splits": {
            "train": args.train_path,
            "val": args.val_path,
            "test": args.test_path,
        },
    }

    if args.mode in {"tfidf", "both"}:
        vectorizer = build_tfidf_vectorizer(
            tokenized_texts=train_texts.tolist(),
            ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
            min_df=int(args.tfidf_min_df) if args.tfidf_min_df >= 1 else args.tfidf_min_df,
            max_df=args.tfidf_max_df,
            max_features=args.tfidf_max_features,
        )
        save_tfidf_vectorizer(vectorizer, artifact_dir / "tfidf_vectorizer.joblib")

        x_train = transform_tfidf(train_texts.tolist(), vectorizer)
        save_sparse_matrix(x_train, output_dir / "X_train_tfidf.npz")

        if val_texts is not None:
            x_val = transform_tfidf(val_texts.tolist(), vectorizer)
            save_sparse_matrix(x_val, output_dir / "X_val_tfidf.npz")

        if test_texts is not None:
            x_test = transform_tfidf(test_texts.tolist(), vectorizer)
            save_sparse_matrix(x_test, output_dir / "X_test_tfidf.npz")

        metadata["tfidf"] = {
            "ngram_range": [args.tfidf_ngram_min, args.tfidf_ngram_max],
            "min_df": args.tfidf_min_df,
            "max_df": args.tfidf_max_df,
            "max_features": args.tfidf_max_features,
            "vocab_size": len(vectorizer.vocabulary_),
        }

    if args.mode in {"fasttext", "both"}:
        ft_model = train_fasttext_model(
            tokenized_texts=train_texts.tolist(),
            vector_size=args.ft_vector_size,
            window=args.ft_window,
            min_count=args.ft_min_count,
            epochs=args.ft_epochs,
            sg=args.ft_sg,
            workers=args.ft_workers,
            seed=args.ft_seed,
        )
        save_fasttext_model(ft_model, artifact_dir / "fasttext.model")

        x_train_ft = documents_to_fasttext_vectors(train_texts.tolist(), ft_model, pooling=args.ft_pooling)
        save_dense_array(x_train_ft, output_dir / "X_train_fasttext.npy")

        if val_texts is not None:
            x_val_ft = documents_to_fasttext_vectors(
                val_texts.tolist(),
                ft_model,
                pooling=args.ft_pooling,
            )
            save_dense_array(x_val_ft, output_dir / "X_val_fasttext.npy")

        if test_texts is not None:
            x_test_ft = documents_to_fasttext_vectors(
                test_texts.tolist(),
                ft_model,
                pooling=args.ft_pooling,
            )
            save_dense_array(x_test_ft, output_dir / "X_test_fasttext.npy")

        metadata["fasttext"] = {
            "vector_size": args.ft_vector_size,
            "window": args.ft_window,
            "min_count": args.ft_min_count,
            "epochs": args.ft_epochs,
            "sg": args.ft_sg,
            "pooling": args.ft_pooling,
        }

    _save_metadata(metadata, output_dir)
    print(f"Saved artifacts to: {artifact_dir}")
    print(f"Saved feature matrices to: {output_dir}")


if __name__ == "__main__":
    main()

