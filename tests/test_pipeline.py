import numpy as np
import torch

from src.data.preprocess import TextPreprocessor, clean_text, expand_abbreviations
from src.features.sequence import Vocab, pad_truncate
from src.models.networks import BiLSTMClassifier, LSTMClassifier, TextCNN

CFG = {"lowercase": True, "remove_urls": True, "remove_html": True, "remove_special_chars": True,
       "normalize_unicode": True, "expand_abbreviations": True, "remove_stopwords": True,
       "tokenizer": "underthesea"}


def test_clean_text_removes_url_and_symbols():
    out = clean_text("Tin nóng!!! Chính phủ công bố chính sách mới tại https://example.com", CFG)
    assert out == "tin nóng chính phủ công bố chính sách mới tại"


def test_expand_abbreviations():
    m = {"bn": "bạn", "ko": "không", "bít": "biết"}
    assert expand_abbreviations("bn ko bít chuyện này hả", m) == "bạn không biết chuyện này hả"


def test_preprocessor_tokenizes_compound_words():
    pre = TextPreprocessor(CFG, {}, {"và", "là"})
    out = pre("Chính phủ công bố chính sách mới")
    assert "chính_phủ" in out.split()


def test_vocab_and_padding():
    v = Vocab.build([["a", "b"], ["a", "c"]])
    ids = v.encode(["a", "zzz"])
    assert ids[1] == v.stoi["<UNK>"]
    arr = pad_truncate([[2, 3, 4, 5], [2]], max_len=3)
    assert arr.shape == (2, 3) and arr[0].tolist() == [2, 3, 4] and arr[1].tolist() == [2, 0, 0]


def test_models_forward_shapes():
    x = torch.randint(1, 50, (4, 20))
    x[0, 10:] = 0
    for m in (TextCNN(50, 16), LSTMClassifier(50, 16, hidden_size=8), BiLSTMClassifier(50, 16, hidden_size=8)):
        assert m(x).shape == (4,)
