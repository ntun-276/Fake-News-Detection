# Fake News Detection - Data Cleaning

This project now includes a text cleaning pipeline for section **2.3.1 Data Cleaning**.

## Implemented cleaning steps

- Remove URLs
- Remove HTML tags
- Remove special characters (keeps Vietnamese letters)
- Remove numbers (configurable)
- Convert text to lowercase
- Remove extra whitespace

## Main files

- `src/data/preprocess.py`: text-level cleaning functions
- `src/data/load_data.py`: dataset loading and cleaning helpers
- `main.py`: CLI to clean CSV and export output
- `scripts/smoke_test_preprocess.py`: quick smoke test

## Quick start

```powershell
python -m pip install -r requirements.txt
python scripts/smoke_test_preprocess.py
python main.py --input data/update_data/update_train_data.csv --output data/update_data/update_train_data_cleaned.csv --text-col Maintext
```

## Notes

- Built-in Python modules (`re`, `string`) are **not** listed in `requirements.txt`.
- Third-party packages used: `pandas`, `beautifulsoup4`.

