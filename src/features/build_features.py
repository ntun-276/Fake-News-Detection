from collections import Counter
from typing import Dict, Iterable, List, Optional

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _split_tokens(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def build_vocabulary(
    tokenized_texts: Iterable[str],
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text in tokenized_texts:
        counter.update(_split_tokens(text))

    sorted_tokens = sorted(
        ((token, freq) for token, freq in counter.items() if freq >= min_freq),
        key=lambda item: (-item[1], item[0]),
    )
    if max_vocab_size is not None:
        sorted_tokens = sorted_tokens[: max(0, max_vocab_size - 2)]

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in sorted_tokens:
        vocab[token] = len(vocab)
    return vocab


def texts_to_sequences(tokenized_texts: Iterable[str], vocab: Dict[str, int]) -> List[List[int]]:
    unk_id = vocab[UNK_TOKEN]
    sequences: List[List[int]] = []
    for text in tokenized_texts:
        sequences.append([vocab.get(token, unk_id) for token in _split_tokens(text)])
    return sequences


def pad_or_truncate_sequences(
    sequences: Iterable[List[int]],
    max_length: int,
    padding: str = "post",
    truncating: str = "post",
    pad_value: int = 0,
) -> List[List[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be either 'pre' or 'post'")
    if truncating not in {"pre", "post"}:
        raise ValueError("truncating must be either 'pre' or 'post'")

    padded_sequences: List[List[int]] = []
    for sequence in sequences:
        current = list(sequence)

        if len(current) > max_length:
            current = current[-max_length:] if truncating == "pre" else current[:max_length]

        pad_len = max_length - len(current)
        if pad_len > 0:
            pads = [pad_value] * pad_len
            current = pads + current if padding == "pre" else current + pads

        padded_sequences.append(current)

    return padded_sequences


