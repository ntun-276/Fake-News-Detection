# Fake News Detection - Huong dan test end-to-end

README nay uu tien de **chay thu tu dau den cuoi**:

1. Tien xu ly van ban (`main.py`)
2. Tao dac trung (`scripts/build_text_features.py`)
3. Kiem tra output cua TF-IDF va FastText

## 1) Yeu cau

- Python 3.10+ (khuyen nghi)
- Windows PowerShell
- Dang o root project `Fake-News-Detection`

## 2) Cai dat

```powershell
python -m pip install -r requirements.txt
```

## 3) Chay smoke test nhanh

```powershell
python scripts/smoke_test_preprocess.py
python scripts/smoke_test_features.py
```

Neu ca hai lenh deu in `passed` thi pipeline co ban da on.

## 4) Luong end-to-end (1 file, test nhanh)

### Buoc A - Tien xu ly tu file goc

Input goc: `data/update_data/fix_test_data.csv` (co cot `Maintext`)

```powershell
python main.py --input data/update_data/fix_test_data.csv --output data/update_data/fix_test_data_preprocessed_tmp.csv --text-col Maintext --output-col clean_text --normalized-col normalized_text --tokenized-col tokenized_text --sequence-col input_ids --max-length 64 --padding post --truncating post --abbreviation-csv data/vietnamese_abbreviation_normalization.csv
```

Sau buoc nay, file `data/update_data/fix_test_data_preprocessed_tmp.csv` se co cac cot:

- `Maintext` (goc)
- `clean_text`
- `normalized_text`
- `tokenized_text`
- `input_ids`

### Buoc B - Tao TF-IDF + FastText tu `tokenized_text`

> Luu y: pipeline feature hien tai chi chap nhan cot `tokenized_text`.

```powershell
python scripts/build_text_features.py --train-path data/update_data/fix_test_data_preprocessed_tmp.csv --tokenized-col tokenized_text --mode both --artifact-dir artifacts/features_fix_test_e2e --output-dir data/features_fix_test_e2e --tfidf-min-df 1 --tfidf-max-df 1.0 --ft-min-count 1
```

## 5) Output sinh ra la gi?

### 5.1 Thu muc artifacts (`artifacts/features_fix_test_e2e`)

- `tfidf_vectorizer.joblib`: model TF-IDF da fit tren train
- `fasttext.model`: model FastText da train
- `fasttext.model.wv.vectors_ngrams.npy`: ma tran subword vectors cua FastText (thuong lon)

### 5.2 Thu muc feature data (`data/features_fix_test_e2e`)

- `X_train_tfidf.npz`: ma tran sparse TF-IDF, shape `(so_van_ban, so_dac_trung)`
- `X_train_fasttext.npy`: ma tran dense FastText, shape `(so_van_ban, vector_size)`
- `feature_metadata.json`: cau hinh lan chay (input, hyper-params, vocab size)

## 6) Y nghia TF-IDF va FastText trong project

- **TF-IDF**: bien moi van ban thanh vector sparse theo tan suat + do hiem cua tu/ngram.
- **FastText**: hoc embedding tu, sau do pooling (mean/sum) thanh vector dense cho moi van ban.
- Hai output nay la **dau vao cho model phan loai** o buoc train/predict.
- Ban than buoc 2.4 khong tra nhan fake/real; no chi tra vector so.

## 7) Kiem tra nhanh output bang Python

```powershell
python -c "import json, numpy as np; from scipy import sparse; md=json.load(open('data/features_fix_test_e2e/feature_metadata.json', encoding='utf-8')); X1=sparse.load_npz('data/features_fix_test_e2e/X_train_tfidf.npz'); X2=np.load('data/features_fix_test_e2e/X_train_fasttext.npy'); print(md['tokenized_col'], md['mode']); print('TFIDF', X1.shape, 'nnz=', X1.nnz); print('FASTTEXT', X2.shape, X2.dtype)"
```

## 8) Chay theo train/val/test (khi co san 3 file da preprocess)

Tat ca file dau vao phai co cot `tokenized_text`.

```powershell
python scripts/build_text_features.py --train-path data/update_data/update_train_data_cleaned.csv --val-path data/update_data/update_val_data.csv --test-path data/update_data/fix_test_data_preprocessed_tmp.csv --tokenized-col tokenized_text --mode both --artifact-dir artifacts/features --output-dir data/features
```

Se sinh them:

- `X_val_tfidf.npz`, `X_test_tfidf.npz`
- `X_val_fasttext.npy`, `X_test_fasttext.npy`

## 9) Loi thuong gap

- `Missing required column 'tokenized_text'`: file dau vao chua qua buoc preprocess hoac sai cot.
- File FastText qua lon: giam `--ft-min-count`, `--ft-vector-size` hoac huan luyen tren tap train gon hon.
- Loi duong dan Windows: dung duong dan tu root project, tranh go sai ten thu muc.

## 10) File chinh trong codebase

- `main.py`: CLI tien xu ly CSV
- `src/data/load_data.py`: pipeline xu ly theo dataset
- `src/data/preprocess.py`: clean + normalize + tokenize
- `src/features/build_features.py`: vocab + sequence + pad/truncate
- `src/features/tfidf_features.py`: fit/transform TF-IDF
- `src/features/fasttext_features.py`: train FastText + vector hoa van ban
- `src/features/artifact_io.py`: save/load vectorizer, model, matrix
- `scripts/build_text_features.py`: tao dac trung 2.4 tu `tokenized_text`
- `scripts/smoke_test_preprocess.py`: smoke test preprocess
- `scripts/smoke_test_features.py`: smoke test feature
