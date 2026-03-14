import re
import string
import unicodedata
from html import unescape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import pandas as pd
from bs4 import BeautifulSoup

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EXTRA_SPACES_PATTERN = re.compile(r"\s+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^0-9A-Za-zÀ-ỹ\s]")
NUMBER_PATTERN = re.compile(r"\d+")
MULTI_PUNCT_PATTERN = re.compile(r"([!?.,:;])\1+")

try:
    from underthesea import word_tokenize as uts_word_tokenize
except ImportError:  # pragma: no cover - fallback is tested via smoke test
    uts_word_tokenize = None

TokenizerCallable = Optional[Callable[..., str]]


def normalize_text_input(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    return str(text)


def remove_url(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(" ")


def remove_special_characters(text: str) -> str:
    # Drop ASCII punctuation quickly, then keep Vietnamese letters via regex.
    text = text.translate(str.maketrans("", "", string.punctuation))
    return SPECIAL_CHAR_PATTERN.sub(" ", text)


def remove_numbers(text: str) -> str:
    return NUMBER_PATTERN.sub(" ", text)


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_extra_spaces(text: str) -> str:
    return EXTRA_SPACES_PATTERN.sub(" ", text).strip()


def clean_text(text: Any, remove_digits: bool = True) -> str:
    text = normalize_text_input(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_special_characters(text)
    if remove_digits:
        text = remove_numbers(text)
    text = to_lowercase(text)
    text = remove_extra_spaces(text)
    return text


def clean_text_series(series: pd.Series, remove_digits: bool = True) -> pd.Series:
    return series.apply(lambda value: clean_text(value, remove_digits=remove_digits))


def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def load_abbreviation_map(csv_path: Optional[Union[str, Path]]) -> Dict[str, str]:
    if not csv_path:
        return {}

    file_path = Path(csv_path)
    if not file_path.exists():
        return {}

    df = pd.read_csv(file_path)
    if df.empty or len(df.columns) < 2:
        return {}

    from_col = _resolve_col(df, ["abbr", "abbreviation", "viet_tat", "short", "from"])
    to_col = _resolve_col(df, ["normalized", "full", "standard", "to", "chuan_hoa"])
    if from_col is None or to_col is None:
        from_col, to_col = df.columns[0], df.columns[1]

    mapping: Dict[str, str] = {}
    for source, target in zip(df[from_col], df[to_col]):
        if pd.isna(source) or pd.isna(target):
            continue
        source_text = normalize_text_input(source).strip().lower()
        target_text = normalize_text_input(target).strip().lower()
        if source_text and target_text:
            mapping[source_text] = target_text
    return mapping


def load_stopwords(path: Optional[Union[str, Path]]) -> Set[str]:
    if not path:
        return set()

    file_path = Path(path)
    if not file_path.exists():
        return set()

    stopwords: Set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if token:
            stopwords.add(token)
    return stopwords


def normalize_vietnamese_text(
    text: Any,
    abbreviation_map: Optional[Dict[str, str]] = None,
    stopwords: Optional[Set[str]] = None,
) -> str:
    text = normalize_text_input(text)
    text = unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = MULTI_PUNCT_PATTERN.sub(r"\1", text)
    text = remove_extra_spaces(text)

    tokens = text.split()
    if abbreviation_map:
        tokens = [abbreviation_map.get(token, token) for token in tokens]
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    return " ".join(tokens)


def normalize_vietnamese_series(
    series: pd.Series,
    abbreviation_map: Optional[Dict[str, str]] = None,
    stopwords: Optional[Set[str]] = None,
) -> pd.Series:
    return series.apply(
        lambda value: normalize_vietnamese_text(
            value,
            abbreviation_map=abbreviation_map,
            stopwords=stopwords,
        )
    )


def tokenize_vietnamese_text(text: Any, join_tokens: bool = True) -> Union[str, List[str]]:
    normalized_text = normalize_vietnamese_text(text)
    if not normalized_text:
        return ""

    tokenizer: TokenizerCallable = uts_word_tokenize
    if callable(tokenizer):
        tokenized = tokenizer(normalized_text, format="text")
    else:
        tokenized = " ".join(normalized_text.split())

    return tokenized if join_tokens else tokenized.split()


def tokenize_vietnamese_series(series: pd.Series) -> pd.Series:
    return series.apply(tokenize_vietnamese_text)
