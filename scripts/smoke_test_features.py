import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.artifact_io import (
    load_fasttext_model,
    load_sparse_matrix,
    load_tfidf_vectorizer,
    save_fasttext_model,
    save_sparse_matrix,
    save_tfidf_vectorizer,
)
from src.features.fasttext_features import documents_to_fasttext_vectors, train_fasttext_model
from src.features.tfidf_features import build_tfidf_vectorizer, transform_tfidf


def run_smoke_test() -> None:
    train_texts = [
        "tin nong gia",
        "tin that chinh thong",
        "bao cao khoa hoc",
        "thong tin chinh xac",
    ]
    test_texts = ["tin nong", "du lieu moi"]

    vectorizer = build_tfidf_vectorizer(train_texts, ngram_range=(1, 2), min_df=1, max_df=1.0)
    x_train = transform_tfidf(train_texts, vectorizer)
    x_test = transform_tfidf(test_texts, vectorizer)

    assert x_train.shape[0] == len(train_texts), "TF-IDF train rows mismatch"
    assert x_test.shape[0] == len(test_texts), "TF-IDF test rows mismatch"
    assert x_train.shape[1] > 0, "TF-IDF vocab size must be > 0"

    ft_model = train_fasttext_model(
        train_texts,
        vector_size=20,
        window=3,
        min_count=1,
        epochs=5,
        workers=1,
    )
    doc_vectors = documents_to_fasttext_vectors(test_texts, ft_model, pooling="mean")

    assert doc_vectors.shape == (2, 20), "FastText output shape mismatch"
    assert np.isfinite(doc_vectors).all(), "FastText vectors contain invalid values"

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        save_tfidf_vectorizer(vectorizer, tmp_path / "tfidf_vectorizer.joblib")
        save_sparse_matrix(x_train, tmp_path / "x_train_tfidf.npz")
        save_fasttext_model(ft_model, tmp_path / "fasttext.model")

        loaded_vectorizer = load_tfidf_vectorizer(tmp_path / "tfidf_vectorizer.joblib")
        loaded_sparse = load_sparse_matrix(tmp_path / "x_train_tfidf.npz")
        loaded_ft_model = load_fasttext_model(tmp_path / "fasttext.model")

        re_x_test = transform_tfidf(test_texts, loaded_vectorizer)
        re_doc_vectors = documents_to_fasttext_vectors(test_texts, loaded_ft_model)

        assert re_x_test.shape == x_test.shape, "Loaded TF-IDF vectorizer does not match shape"
        assert loaded_sparse.shape == x_train.shape, "Loaded sparse matrix shape mismatch"
        assert re_doc_vectors.shape == doc_vectors.shape, "Loaded FastText model shape mismatch"

    print("Feature smoke test passed")


if __name__ == "__main__":
    run_smoke_test()


