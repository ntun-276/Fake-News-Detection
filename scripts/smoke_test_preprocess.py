import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import clean_dataset
from src.data.preprocess import (
    clean_text,
    load_abbreviation_map,
    normalize_vietnamese_text,
    tokenize_vietnamese_text,
)


def run_smoke_test() -> None:
    sample = "<p>Tin nóng!!! Xem tại https://example.com vào 2026 nhé :)</p>"
    cleaned = clean_text(sample)

    expected = "tin nóng xem tại vào nhé"
    assert cleaned == expected, f"Expected: '{expected}', got: '{cleaned}'"

    normalized = normalize_vietnamese_text("Toi&#39; dang hoc AI!!!")
    assert normalized == "Toi' dang hoc AI!", f"Unexpected normalization output: '{normalized}'"

    tokenized = tokenize_vietnamese_text("thành phố hồ chí minh")
    assert isinstance(tokenized, str) and tokenized.strip(), "Tokenization should return non-empty text"

    abbreviation_map = load_abbreviation_map(PROJECT_ROOT / "data" / "vietnamese_abbreviation_normalization.csv")
    normalized_short = normalize_vietnamese_text("ko j", abbreviation_map=abbreviation_map)
    assert normalized_short == "không gì", f"Unexpected abbreviation output: '{normalized_short}'"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_file:
        temp_file.write("tin\n")
        stopword_file = Path(temp_file.name)

    df = clean_dataset(
        df=pd.DataFrame({"Maintext": ["ko j tin", "Bản tin chính thống"]}),
        max_length=5,
        abbreviation_csv=PROJECT_ROOT / "data" / "vietnamese_abbreviation_normalization.csv",
        stopwords_path=stopword_file,
    )
    assert "input_ids" in df.columns, "Missing padded sequence column"
    assert all(len(seq) == 5 for seq in df["input_ids"]), "All sequences must have fixed max_length"
    assert df.loc[0, "normalized_text"] == "không gì", "Stopword removal or abbreviation normalization failed"

    stopword_file.unlink(missing_ok=True)
    print("Smoke test passed")


if __name__ == "__main__":
    run_smoke_test()
