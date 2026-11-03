"""Bước 4: gộp metrics của các mô hình thành bảng so sánh (reports/comparison.csv)."""
from __future__ import annotations

import json

import pandas as pd

from src.utils.config import load_config, resolve


def main() -> None:
    cfg = load_config()
    rep = resolve(cfg, "reports_dir")
    rows = []
    for f in sorted(rep.glob("metrics_*.json")):
        m = json.loads(f.read_text())
        rows.append({"model": f.stem.replace("metrics_", ""),
                     **{k: round(m[k], 4) for k in ("accuracy", "precision", "recall", "f1")}})
    if not rows:
        raise SystemExit("Chưa có reports/metrics_*.json - hãy chạy scripts/train.py trước.")
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(rep / "comparison.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
