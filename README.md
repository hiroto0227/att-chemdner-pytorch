# predict-template

## how to
```
# rawデータを学習可能なデータに変換する。
>>> python scripts/data/processed.py

# modelを学習する。
>>> python scripts/model/train.py

# modelを評価する。
>>> python scripts/evaluate.py
```

## directory Architecture
- datas
  - raw (生データを置く)
  - processed (学習しやすい形に変形したデータを格納)
- models (modelのparameterやtransformerを保存する。)
- scripts
  - data
    - split.py (train_testを分ける。)
    - processed.py (model.trainで学習しやすい形に変形し、datas/processedにsave)
  - model
    - train.py
  - evaluate.py
