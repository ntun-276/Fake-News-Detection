"""Tải dữ liệu thô và chia train/val/test."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def find_raw_file(raw_dir: Path, raw_file: str | None = None) -> Path:
    if raw_file:
        p = raw_dir / raw_file
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = sorted(raw_dir.glob("**/*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Không thấy file .csv trong {raw_dir}. Chạy `python scripts/download_data.py` "
            "hoặc đặt file dataset vào data/raw/."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Có {len(candidates)} file CSV trong {raw_dir}: {[c.name for c in candidates]}. "
            "Hãy chỉ định dataset.raw_file trong config (hoặc dùng split_mode: presplit)."
        )
    return candidates[0]


def _read_csv(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise KeyError(f"Thiếu cột {missing} trong {path.name}. Các cột hiện có: {list(df.columns)}")
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)
    return df


def load_presplit(raw_dir: Path, ds: dict):
    """Dataset PBL7 đã chia sẵn train/val/test. Chỉ bỏ bài trùng trong từng tập và bài của train
    trùng với val/test (chống rò rỉ); val/test giữ nguyên."""
    text_col, label_col = ds["text_column"], ds["label_column"]
    train = _read_csv(raw_dir / ds["train_file"], text_col, label_col)
    val = _read_csv(raw_dir / ds["val_file"], text_col, label_col)
    test = _read_csv(raw_dir / ds["test_file"], text_col, label_col)
    strip = lambda d: d[text_col].str.strip()
    train = train[~strip(train).duplicated()]
    held_out = set(strip(val)) | set(strip(test))
    train = train[~strip(train).isin(held_out)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_raw(raw_dir: Path, text_col: str, label_col: str, raw_file: str | None = None) -> pd.DataFrame:
    path = find_raw_file(raw_dir, raw_file)
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise KeyError(f"Thiếu cột {missing} trong {path.name}. Các cột hiện có: {list(df.columns)}")
    df = df[[text_col, label_col]].dropna().drop_duplicates().reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)
    return df


def split_dataset(df: pd.DataFrame, label_col: str, test_size: float, val_size: float, seed: int):
    """Chia phân tầng (stratified) theo nhãn: train / val / test."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val[label_col], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
