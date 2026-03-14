import re
import string
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EXTRA_SPACES_PATTERN = re.compile(r"\s+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^0-9A-Za-zÀ-ỹ\s]")
NUMBER_PATTERN = re.compile(r"\d+")


def normalize_text_input(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    return str(text)


def remove_url(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()


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
