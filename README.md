# Fake News Detection - Huong dan chay lai tu dau den cuoi

README nay viet lai de de hieu va de test, tap trung vao 3 phan:

1. Tien xu ly va tao `input_ids`
2. Bieu dien du lieu TF-IDF + FastText (tu `tokenized_text`)
3. Train va so sanh LSTM vs BiLSTM 100 epoch (bat buoc dung `input_ids`)

## 1) Yeu cau

- Python 3.10+
- Windows PowerShell
- Dang o root: `D:\ĐACN\Fake-News-Detection`

## 2) Cai dat

```powershell
python -m pip install -r requirements.txt
```

## 3) Kiem tra nhanh pipeline

```powershell
python scripts/smoke_test_preprocess.py
python scripts/smoke_test_features.py
python scripts/smoke_test_lstm.py
python scripts/smoke_test_bilstm.py
```

Neu 4 lenh deu in `passed` thi code co ban dang chay duoc.

## 4) Luong du lieu trong project

### 4.1 Luong preprocess

Input goc: cot `Maintext`

Output sau preprocess:

- `clean_text`
- `normalized_text`
- `tokenized_text`
- `input_ids`

Trong project hien tai:

- TF-IDF va FastText dung `tokenized_text`
- LSTM va BiLSTM dung `input_ids` (khong fallback ve cot khac)

### 4.2 Luu y quan trong cho train deep learning

`scripts/train_lstm_baseline.py` va `scripts/train_bilstm_phase2.py` da khoa theo input_ids-only:

- Tat ca split train/val/test ban dua vao deu phai co cot `input_ids`
- Neu thieu `input_ids`, script se bao loi va dung

## 5) Tao file input_ids cho train/val de train cong bang

Hai file goc sau thuong chua co `input_ids`:

- `data/update_data/update_train_data.csv`
- `data/update_data/update_val_data.csv`

Lenh duoi day tao 2 file moi co `input_ids`, va dung chung vocabulary tu train cho val:

```powershell
python -u -c "from pathlib import Path; import pandas as pd; from src.data.load_data import clean_dataset; from src.features.build_features import build_vocabulary; train_in=Path('data/update_data/update_train_data.csv'); val_in=Path('data/update_data/update_val_data.csv'); train_out=Path('data/update_data/update_train_input_ids.csv'); val_out=Path('data/update_data/update_val_input_ids.csv'); train_df=pd.read_csv(train_in); val_df=pd.read_csv(val_in); train_clean=clean_dataset(train_df, text_col='Maintext', sequence_col='input_ids', tokenized_col='tokenized_text', normalized_col='normalized_text', output_col='clean_text', max_length=64, min_freq=1, abbreviation_csv='data/vietnamese_abbreviation_normalization.csv'); vocab=build_vocabulary(train_clean['tokenized_text'].fillna('').astype(str).tolist(), min_freq=1, max_vocab_size=None); val_clean=clean_dataset(val_df, text_col='Maintext', sequence_col='input_ids', tokenized_col='tokenized_text', normalized_col='normalized_text', output_col='clean_text', max_length=64, vocab=vocab, abbreviation_csv='data/vietnamese_abbreviation_normalization.csv'); train_out.parent.mkdir(parents=True, exist_ok=True); train_clean.to_csv(train_out, index=False, encoding='utf-8-sig'); val_clean.to_csv(val_out, index=False, encoding='utf-8-sig'); print('saved', train_out); print('saved', val_out)"
```

Sau khi chay, ban co 2 file de train:

- `data/update_data/update_train_input_ids.csv`
- `data/update_data/update_val_input_ids.csv`

## 6) TF-IDF + FastText (phan bieu dien du lieu)

Vi du nhanh tren file da preprocess:

```powershell
python scripts/build_text_features.py --train-path data/update_data/fix_test_data_preprocessed_tmp.csv --tokenized-col tokenized_text --mode both --artifact-dir artifacts/features_fix_test_e2e --output-dir data/features_fix_test_e2e --tfidf-min-df 1 --tfidf-max-df 1.0 --ft-min-count 1
```

Output chinh:

- `artifacts/features_fix_test_e2e/tfidf_vectorizer.joblib`
- `artifacts/features_fix_test_e2e/fasttext.model`
- `data/features_fix_test_e2e/X_train_tfidf.npz`
- `data/features_fix_test_e2e/X_train_fasttext.npy`
- `data/features_fix_test_e2e/feature_metadata.json`

## 7) Train LSTM 100 epoch (Phase 1)

```powershell
python scripts/train_lstm_baseline.py --train-path data/update_data/update_train_input_ids.csv --val-path data/update_data/update_val_input_ids.csv --output-dir artifacts/lstm_phase1_epoch100_inputids --epochs 100 --batch-size 32 --max-length 64 --embedding-dim 128 --hidden-size 128 --num-layers 1 --dropout 0.2 --learning-rate 1e-3 --seed 42
```

Output:

- `artifacts/lstm_phase1_epoch100_inputids/lstm_baseline.pt`
- `artifacts/lstm_phase1_epoch100_inputids/metrics.json`

## 8) Train BiLSTM 100 epoch (Phase 2)

```powershell
python scripts/train_bilstm_phase2.py --train-path data/update_data/update_train_input_ids.csv --val-path data/update_data/update_val_input_ids.csv --output-dir artifacts/bilstm_phase2_epoch100_inputids --epochs 100 --batch-size 32 --max-length 64 --embedding-dim 128 --hidden-size 128 --num-layers 1 --dropout 0.2 --learning-rate 1e-3 --seed 42
```

Output:

- `artifacts/bilstm_phase2_epoch100_inputids/bilstm_phase2.pt`
- `artifacts/bilstm_phase2_epoch100_inputids/metrics.json`

## 9) So sanh LSTM va BiLSTM

### 9.1 So sanh nhanh bang lenh 1 dong

```powershell
python -u -c "import json; l=json.load(open('artifacts/lstm_phase1_epoch100_inputids/metrics.json',encoding='utf-8')); b=json.load(open('artifacts/bilstm_phase2_epoch100_inputids/metrics.json',encoding='utf-8')); lf=l['history'][-1]; bf=b['history'][-1]; lb=max(l['history'], key=lambda x: x.get('val_f1',-1)); bb=max(b['history'], key=lambda x: x.get('val_f1',-1)); print('FINAL LSTM  :', lf); print('FINAL BiLSTM:', bf); print('BEST LSTM   :', lb); print('BEST BiLSTM :', bb)"
```

### 9.2 Hieu cho dung ket qua

- `FINAL`: metric tai epoch 100
- `BEST`: epoch co `val_f1` cao nhat trong 100 epoch
- De bao cao cong bang, nen ghi ca `FINAL` va `BEST`

## 10) File quan trong trong project

- `main.py`: preprocess CSV
- `scripts/build_text_features.py`: tao TF-IDF + FastText tu `tokenized_text`
- `scripts/train_lstm_baseline.py`: train LSTM
- `scripts/train_bilstm_phase2.py`: train BiLSTM
- `scripts/smoke_test_preprocess.py`: smoke preprocess
- `scripts/smoke_test_features.py`: smoke features
- `scripts/smoke_test_lstm.py`: smoke LSTM
- `scripts/smoke_test_bilstm.py`: smoke BiLSTM

## 11) Loi thuong gap va cach xu ly

- Loi `Missing required input_ids...`: ban dang dua file chua co `input_ids` vao script train.
- Train ra `sequence_mode` khong phai `input_ids`: can kiem tra lai file dau vao, hien tai script da khoa input_ids-only.
- FastText train lau/ton RAM: giam `--ft-vector-size`, tang `--ft-min-count`, hoac giam kich thuoc tap train.
- Sai duong dan: luon chay lenh tai root `Fake-News-Detection`.
