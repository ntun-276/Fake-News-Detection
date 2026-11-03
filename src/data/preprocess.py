"""Tiền xử lý văn bản tiếng Việt: làm sạch -> chuẩn hoá -> tách từ -> bỏ stopword."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
# giữ chữ cái (kể cả có dấu tiếng Việt), chữ số và khoảng trắng
SPECIAL_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def load_abbreviations(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return {str(a).strip().lower(): str(n).strip() for a, n in zip(df.iloc[:, 0], df.iloc[:, 1])}


def load_stopwords(path: Path) -> set[str]:
    return {w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}


def clean_text(text: str, cfg: dict) -> str:
    if cfg.get("normalize_unicode", True):
        text = unicodedata.normalize("NFC", text)
    if cfg.get("remove_urls", True):
        text = URL_RE.sub(" ", text)
    if cfg.get("remove_html", True):
        text = HTML_RE.sub(" ", text)
    if cfg.get("lowercase", True):
        text = text.lower()
    if cfg.get("remove_special_chars", True):
        text = SPECIAL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def expand_abbreviations(text: str, mapping: dict[str, str]) -> str:
    return " ".join(mapping.get(tok, tok) for tok in text.split())


def tokenize(text: str, method: str = "underthesea") -> list[str]:
    if method == "underthesea":
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").split()
    return text.split()


class TextPreprocessor:
    def __init__(self, cfg: dict, abbreviations: dict[str, str], stopwords: set[str]):
        self.cfg = cfg
        self.abbr = abbreviations
        self.stop = stopwords

    def __call__(self, text: str) -> str:
        text = clean_text(str(text), self.cfg)
        if self.cfg.get("expand_abbreviations", True):
            text = expand_abbreviations(text, self.abbr)
        tokens = tokenize(text, self.cfg.get("tokenizer", "underthesea"))
        if self.cfg.get("remove_stopwords", True):
            tokens = [t for t in tokens if t not in self.stop]
        return " ".join(tokens)

    def transform_series(self, s: pd.Series, show_progress: bool = True) -> pd.Series:
        if show_progress:
            from tqdm import tqdm

            tqdm.pandas(desc="preprocess")
            return s.progress_apply(self)
        return s.apply(self)
