import os, sys
import torch
import torchtext
from torchtext.data import Iterator
from dataset import ChemdnerDataset

from seqeval.metrics import f1_score
from seqeval.metrics.sequence_labeling import get_entities

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    fields = [('text', train_dataset.text_field), ('label', train_dataset.label_field)]
    test_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/test.csv'), fields=fields)

    model = load_model()
    all_true_labels = []
    all_pred_labels = []
    for batch in Iterator(test_dataset, batch_size=1, shuffle=False, repeat=False):
        tokens = [id2token[b] for b in batch.text.transpose(1, 0)[0]]
        true_labels = [id2label[b] for b in batch.label.transpose(1, 0)[0]]
        output = model(batch.text)
        pred_labels = [id2label[b] for b in output]

        for _, start_idx, end_idx in get_entities(true_labels):
            print(''.join(tokens[start_idx:end_idx + 1]))

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
    print('f1-score: {}'.format(f1_score(all_true_labels, all_pred_labels)))