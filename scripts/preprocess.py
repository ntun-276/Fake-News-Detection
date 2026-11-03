"""Bước 1: EDA + tiền xử lý (+ chia train/val/test nếu split_mode = random).

Đầu ra:
  data/interim/cleaned.csv           toàn bộ dữ liệu đã xử lý (cột tokenized_text, cột split)
  data/processed/{train,val,test}.csv
  reports/figures/label_distribution.png, text_length_distribution.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_presplit, load_raw, split_dataset
from src.data.preprocess import TextPreprocessor, load_abbreviations, load_stopwords
from src.evaluation.plots import plot_label_distribution, plot_length_distribution
from src.utils.config import load_config, resolve, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default=None, help="(split_mode=random) đường dẫn CSV")
    ap.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N dòng mỗi tập (để chạy thử)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ds = cfg["dataset"]
    text_col, label_col, tok_col = ds["text_column"], ds["label_column"], ds["tokenized_column"]
    raw_dir = resolve(cfg, "raw_dir")

    if ds.get("split_mode", "random") == "presplit":
        parts = dict(zip(("train", "val", "test"), load_presplit(raw_dir, ds)))
    else:
        if args.input:
            raw_dir, raw_file = Path(args.input).parent, Path(args.input).name
        else:
            raw_file = ds["raw_file"]
        df = load_raw(raw_dir, text_col, label_col, raw_file)
        parts = dict(zip(("train", "val", "test"),
                         split_dataset(df, label_col, ds["test_size"], ds["val_size"], cfg["seed"])))

    if args.limit:
        parts = {n: d.groupby(label_col, group_keys=False).head(max(args.limit // 2, 1)).reset_index(drop=True)
                 for n, d in parts.items()}

    for n, d in parts.items():
        print(f"{n:5s}: {len(d):4d} bài | nhãn {d[label_col].value_counts().sort_index().to_dict()}")

    all_raw = pd.concat([d.assign(split=n) for n, d in parts.items()], ignore_index=True)
    fig_dir = resolve(cfg, "reports_dir") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_label_distribution(all_raw[label_col], fig_dir / "label_distribution.png")
    plot_length_distribution(all_raw[text_col].str.split().str.len(), fig_dir / "text_length_distribution.png")

    pre = TextPreprocessor(
        cfg["preprocess"],
        load_abbreviations(resolve(cfg, "abbreviation_file")),
        load_stopwords(resolve(cfg, "stopwords_file")),
    )
    out = resolve(cfg, "processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for n, d in parts.items():
        d = d.copy()
        d[tok_col] = pre.transform_series(d[text_col])
        d = d[d[tok_col].str.len() > 0].reset_index(drop=True)
        d[[tok_col, label_col]].to_csv(out / f"{n}.csv", index=False)
        cleaned.append(d[[tok_col, label_col]].assign(split=n))

    interim = resolve(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)
    full = pd.concat(cleaned, ignore_index=True)
    full.to_csv(interim / "cleaned.csv", index=False)

    n_tok = full[tok_col].str.split().str.len()
    print("\nSố token sau tiền xử lý: mean %.0f | median %.0f | p75 %.0f | p90 %.0f | p95 %.0f | max %d"
          % (n_tok.mean(), n_tok.median(), *n_tok.quantile([.75, .9, .95]), n_tok.max()))
    print("-> chọn sequence.max_seq_len trong configs/config.yaml dựa trên các phân vị này")
    print(f"Đã lưu vào {out}")


if __name__ == "__main__":
    main()
