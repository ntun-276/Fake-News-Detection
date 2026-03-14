from __future__ import annotations

from typing import Iterable, Optional, Tuple

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalize_tokenized_texts(tokenized_texts: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for text in tokenized_texts:
        if text is None:
            normalized.append("")
        elif isinstance(text, str):
            normalized.append(" ".join(text.split()))
        else:
            normalized.append(str(text))
    return normalized


def build_tfidf_vectorizer(
    tokenized_texts: Iterable[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    max_df: int | float = 0.95,
    max_features: Optional[int] = None,
) -> TfidfVectorizer:
    corpus = _normalize_tokenized_texts(tokenized_texts)
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    vectorizer.fit(corpus)
    return vectorizer


def transform_tfidf(tokenized_texts: Iterable[str], vectorizer: TfidfVectorizer) -> csr_matrix:
    corpus = _normalize_tokenized_texts(tokenized_texts)
    return vectorizer.transform(corpus)


def fit_transform_tfidf(
    tokenized_texts: Iterable[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    max_df: int | float = 0.95,
    max_features: Optional[int] = None,
) -> tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = build_tfidf_vectorizer(
        tokenized_texts=tokenized_texts,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    matrix = transform_tfidf(tokenized_texts, vectorizer)
    return vectorizer, matrix

