import argparse
from pathlib import Path

from src.data.load_data import load_and_clean_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean text data for fake news detection.")
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
        "--keep-digits",
        action="store_true",
        help="Keep numbers in text instead of removing them.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cleaned_df = load_and_clean_dataset(
        csv_path=Path(args.input),
        text_col=args.text_col,
        output_col=args.output_col,
        remove_digits=not args.keep_digits,
        output_path=Path(args.output),
    )

    print(f"Done: cleaned {len(cleaned_df)} rows")
    print(f"Saved to: {args.output}")

    preview = cleaned_df[[args.text_col, args.output_col]].head(5).copy()
    preview[args.text_col] = preview[args.text_col].astype(str).str.slice(0, 120)
    preview[args.output_col] = preview[args.output_col].astype(str).str.slice(0, 120)
    print("Preview (first 5 rows, truncated to 120 chars):")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()

