from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.data.preprocess import (
    clean_text_series,
    load_abbreviation_map,
    load_stopwords,
    normalize_vietnamese_series,
    tokenize_vietnamese_series,
)
from src.features.build_features import (
    PAD_TOKEN,
    build_vocabulary,
    pad_or_truncate_sequences,
    texts_to_sequences,
)

PathLike = Union[str, Path]


def load_dataset(csv_path: PathLike, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding=encoding)


def clean_dataset(
    df: pd.DataFrame,
    text_col: str = "Maintext",
    output_col: str = "clean_text",
    remove_digits: bool = True,
    normalized_col: str = "normalized_text",
    tokenized_col: str = "tokenized_text",
    sequence_col: str = "input_ids",
    max_length: int = 64,
    padding: str = "post",
    truncating: str = "post",
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    vocab: Optional[Dict[str, int]] = None,
    abbreviation_csv: Optional[PathLike] = "data/vietnamese_abbreviation_normalization.csv",
    stopwords_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")

    result = df.copy()
    result[output_col] = clean_text_series(result[text_col], remove_digits=remove_digits)

    abbreviation_map = load_abbreviation_map(abbreviation_csv)
    stopwords = load_stopwords(stopwords_path)
    result[normalized_col] = normalize_vietnamese_series(
        result[output_col],
        abbreviation_map=abbreviation_map,
        stopwords=stopwords,
    )
    result[tokenized_col] = tokenize_vietnamese_series(result[normalized_col])

    effective_vocab = vocab or build_vocabulary(
        result[tokenized_col].tolist(),
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )
    sequences = texts_to_sequences(result[tokenized_col].tolist(), effective_vocab)
    result[sequence_col] = pad_or_truncate_sequences(
        sequences,
        max_length=max_length,
        padding=padding,
        truncating=truncating,
        pad_value=effective_vocab[PAD_TOKEN],
    )
    return result


def load_and_clean_dataset(
    csv_path: PathLike,
    text_col: str = "Maintext",
    output_col: str = "clean_text",
    remove_digits: bool = True,
    normalized_col: str = "normalized_text",
    tokenized_col: str = "tokenized_text",
    sequence_col: str = "input_ids",
    max_length: int = 64,
    padding: str = "post",
    truncating: str = "post",
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    vocab: Optional[Dict[str, int]] = None,
    abbreviation_csv: Optional[PathLike] = "data/vietnamese_abbreviation_normalization.csv",
    stopwords_path: Optional[PathLike] = None,
    output_path: Optional[PathLike] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    df = load_dataset(csv_path, encoding=encoding)
    cleaned_df = clean_dataset(
        df,
        text_col=text_col,
        output_col=output_col,
        remove_digits=remove_digits,
        normalized_col=normalized_col,
        tokenized_col=tokenized_col,
        sequence_col=sequence_col,
        max_length=max_length,
        padding=padding,
        truncating=truncating,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        vocab=vocab,
        abbreviation_csv=abbreviation_csv,
        stopwords_path=stopwords_path,
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return cleaned_df

