"""Đọc cấu hình YAML và tiện ích đường dẫn."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def resolve(cfg, key):
    """
    Resolve a path from config.yaml relative to project root.
    """
    path = Path(cfg["paths"][key])

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_config(config_path=None):
    """
    Load YAML configuration.
    """

    if config_path is None:
        config_path = "configs/config.yaml"

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ImportError:
        pass
