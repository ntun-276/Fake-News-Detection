import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import clean_text


def run_smoke_test() -> None:
    sample = "<p>Tin nóng!!! Xem tại https://example.com vào 2026 nhé :)</p>"
    cleaned = clean_text(sample)

    expected = "tin nóng xem tại vào nhé"
    assert cleaned == expected, f"Expected: '{expected}', got: '{cleaned}'"
    print("Smoke test passed")


if __name__ == "__main__":
    run_smoke_test()
