# chemdner pytorch

## Pretrain
```
# 大規模コーパスの作成
>>> python3.7 parse_PubMed.py --input-dir . --output-path out.txt

# sentencepieceの学習
>>> python3.7 sp_train.py --input-path ../../../PubMed/out.txt --vocab-size 16000

# fasttextで学習するために大規模コーパスをtokenizeする。
>>> python3.7 tokenize_for_pretrain.py --input-path ../../../PubMed/out.txt --sp-model ../sentencepieces/sp16000.model --output-path tokenized_out.txt

# fasttextで学習する。
>>> fastText-0.1.0/fasttext skipgram -input ./tokenized_out.txt -output pretrain.model
```
