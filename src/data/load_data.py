from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.data.preprocess import clean_text_series

PathLike = Union[str, Path]


def load_dataset(csv_path: PathLike, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding=encoding)


def clean_dataset(
    df: pd.DataFrame,
    text_col: str = "Maintext",
    output_col: str = "clean_text",
    remove_digits: bool = True,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")

    result = df.copy()
    result[output_col] = clean_text_series(result[text_col], remove_digits=remove_digits)
    return result


def load_and_clean_dataset(
    csv_path: PathLike,
    text_col: str = "Maintext",
    output_col: str = "clean_text",
    remove_digits: bool = True,
    output_path: Optional[PathLike] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    df = load_dataset(csv_path, encoding=encoding)
    cleaned_df = clean_dataset(
        df,
        text_col=text_col,
        output_col=output_col,
        remove_digits=remove_digits,
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return cleaned_df

