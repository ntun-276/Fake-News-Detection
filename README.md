# Fake News Detection - Vietnamese Text Preprocessing

This project now supports the full preprocessing flow in section 2.3:

- **2.3.1 Data cleaning**
- **2.3.2 Vietnamese text normalization**
- **2.3.3 Vietnamese tokenization**
- **2.3.4 Padding and truncating**

## What is implemented

### 2.3.1 Data cleaning

- Remove URL and HTML
- Remove punctuation/special characters
- Optional digit removal
- Lowercase and remove extra spaces

### 2.3.2 Vietnamese text normalization

- Decode HTML entities (`&#39;` -> `'`)
- Normalize Unicode form to NFC
- Replace abbreviations from `data/vietnamese_abbreviation_normalization.csv`
- Optional stopword removal from a text file (one word per line)
- Normalize repeated punctuation and extra whitespace

### 2.3.3 Vietnamese tokenization

- Primary tokenizer: `underthesea.word_tokenize(..., format="text")`
- Safe fallback: whitespace tokenization when `underthesea` is unavailable

### 2.3.4 Padding and truncating

- Build vocabulary with special tokens: `<PAD>=0`, `<UNK>=1`
- Convert tokenized text to integer sequences
- Pad/truncate to fixed `max_length` with configurable `pre`/`post`

## Main files

- `src/data/preprocess.py`: cleaning + normalization + tokenization functions
- `src/features/build_features.py`: vocabulary, sequence conversion, padding/truncating
- `src/data/load_data.py`: dataset-level preprocessing pipeline
- `main.py`: CLI for preprocessing CSV files
- `scripts/smoke_test_preprocess.py`: smoke test for all preprocessing steps

## Install

```powershell
python -m pip install -r requirements.txt
```

## Quick smoke test

```powershell
python scripts/smoke_test_preprocess.py
```

## Run preprocessing from CLI

```powershell
python main.py --input data/update_data/update_train_data.csv --output data/update_data/update_train_data_cleaned.csv --text-col Maintext --output-col clean_text --normalized-col normalized_text --tokenized-col tokenized_text --sequence-col input_ids --max-length 64 --padding post --truncating post --abbreviation-csv data/vietnamese_abbreviation_normalization.csv
```

## Output columns

After running preprocessing, output CSV contains:

- Original text column (e.g., `Maintext`)
- `clean_text`: cleaned text
- `normalized_text`: normalized Vietnamese text
- `tokenized_text`: Vietnamese tokenized text
- `input_ids`: fixed-length integer sequence (padded/truncated)

## Key CLI options

- `--keep-digits`: keep numbers in cleaning step
- `--abbreviation-csv`: abbreviation mapping CSV path
- `--stopwords-path`: optional stopword file path
- `--max-length`: target sequence length
- `--padding`: `pre` or `post`
- `--truncating`: `pre` or `post`
- `--min-freq`: minimum token frequency for vocabulary
- `--max-vocab-size`: cap vocabulary size (including `<PAD>`, `<UNK>`)
