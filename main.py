import argparse
import sys
from pathlib import Path

from src.data.load_data import load_and_clean_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese text for fake news detection.")
    parser.add_argument(
        "--input",
        default="data/update_data/update_train_data.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/update_data/update_train_data_cleaned.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--text-col",
        default="Maintext",
        help="Name of text column to clean.",
    )
    parser.add_argument(
        "--output-col",
        default="clean_text",
        help="Name of cleaned text column.",
    )
    parser.add_argument(
        "--normalized-col",
        default="normalized_text",
        help="Name of normalized text column.",
    )
    parser.add_argument(
        "--tokenized-col",
        default="tokenized_text",
        help="Name of tokenized text column.",
    )
    parser.add_argument(
        "--sequence-col",
        default="input_ids",
        help="Name of padded/truncated sequence column.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Fixed sequence length used for padding and truncating.",
    )
    parser.add_argument(
        "--padding",
        choices=["pre", "post"],
        default="post",
        help="Padding side.",
    )
    parser.add_argument(
        "--truncating",
        choices=["pre", "post"],
        default="post",
        help="Truncation side.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum token frequency to keep in vocabulary.",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=None,
        help="Maximum vocabulary size including <PAD> and <UNK>.",
    )
    parser.add_argument(
        "--keep-digits",
        action="store_true",
        help="Keep numbers in text instead of removing them.",
    )
    parser.add_argument(
        "--abbreviation-csv",
        default="data/vietnamese_abbreviation_normalization.csv",
        help="CSV path for abbreviation normalization mappings.",
    )
    parser.add_argument(
        "--stopwords-path",
        default=None,
        help="Optional text file path containing one stopword per line.",
    )
    return parser


def main() -> None:
    # Ignore stray PowerShell line-continuation token accidentally passed to argparse.
    sys.argv = [arg for arg in sys.argv if arg.strip() != "`"]

    parser = build_parser()
    args = parser.parse_args()

    cleaned_df = load_and_clean_dataset(
        csv_path=Path(args.input),
        text_col=args.text_col,
        output_col=args.output_col,
        remove_digits=not args.keep_digits,
        normalized_col=args.normalized_col,
        tokenized_col=args.tokenized_col,
        sequence_col=args.sequence_col,
        max_length=args.max_length,
        padding=args.padding,
        truncating=args.truncating,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        abbreviation_csv=Path(args.abbreviation_csv) if args.abbreviation_csv else None,
        stopwords_path=Path(args.stopwords_path) if args.stopwords_path else None,
        output_path=Path(args.output),
    )

    print(f"Done: cleaned {len(cleaned_df)} rows")
    print(f"Saved to: {args.output}")

    preview = cleaned_df[
        [args.text_col, args.output_col, args.normalized_col, args.tokenized_col, args.sequence_col]
    ].head(5).copy()
    preview[args.text_col] = preview[args.text_col].astype(str).str.slice(0, 120)
    preview[args.output_col] = preview[args.output_col].astype(str).str.slice(0, 120)
    preview[args.normalized_col] = preview[args.normalized_col].astype(str).str.slice(0, 120)
    preview[args.tokenized_col] = preview[args.tokenized_col].astype(str).str.slice(0, 120)
    preview[args.sequence_col] = preview[args.sequence_col].astype(str).str.slice(0, 120)
    print("Preview (first 5 rows, truncated to 120 chars):")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
