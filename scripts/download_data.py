"""Tải dataset Vietnamese Fake News PBL7 từ Kaggle về data/raw/.

Yêu cầu: tài khoản Kaggle + token API (https://www.kaggle.com/settings/api)
    export KAGGLE_API_TOKEN=xxxx      # hoặc lưu vào ~/.kaggle/access_token
hoặc chạy `kaggle auth login`.

Nếu không muốn dùng API: tải thủ công trên Kaggle, giải nén vào data/raw/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from fakenews.utils.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    slug = cfg["dataset"]["kaggle_slug"]

    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir), "--unzip"]
    print("Chạy:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
