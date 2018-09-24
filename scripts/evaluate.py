import os, sys
import torch
import torchtext
from torchtext.data import Iterator
from dataset import ChemdnerDataset
from model.lstm import LSTMTagger
from seqeval.metrics import f1_score
from seqeval.metrics.sequence_labeling import get_entities
from tqdm import tqdm

def evaluate(dataset, model, batch_size, text_field, label_field, id2label, verbose=1):
    all_true_labels = []
    all_pred_labels = []
    model.eval()
    eval_iter = Iterator(dataset, batch_size=batch_size, shuffle=False, repeat=False)
    eval_iter.create_batches()

    for batch in tqdm(eval_iter.batches):
        tokens = [b.text for b in batch]
        texts = text_field.process([b.text for b in batch], device=-1, train=False)
        labels = label_field.process([b.label for b in batch], device=-1, train=False)
        true_labels = [[id2label[label_id] for label_id in batch] for batch in labels.transpose(1, 0)]
        output = model(texts)
        _, label_ids = output.max(2) # (seq_length, batch_size)
        pred_labels = [[id2label[label_id] for label_id in batch] for batch in label_ids.transpose(1, 0)]
        all_true_labels.extend([t for true_label in true_labels for t in true_label])
        all_pred_labels.extend([p for pred_label in pred_labels for p in pred_label])

        if verbose:
            for i, true_label in enumerate(pred_labels):
                #print('\n---------------------\n')
                #print(tokens[i])
                #print('\n---------------------\n')
                #print(true_label)
                for ne_type, start_idx, end_idx in get_entities(true_label):
                    if not ne_type == '<pad>':
                        print(''.join(tokens[i][start_idx:end_idx + 1]))
    return f1_score(all_true_labels, all_pred_labels)

if __name__ == '__main__':
    BATCH_SIZE = 50
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '../models/lstm_1ep_50ba')
    
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2token, id2label = [k for k, v in token2id.items()], [k for k, v in label2id.items()]
    fields = [('text', train_dataset.text_field), ('label', train_dataset.label_field)]
    test_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/test.csv'), fields=fields)
    
    model = LSTMTagger(len(token2id), len(label2id), batch_size=BATCH_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))

    f1score = evaluate(dataset=test_dataset, model=model, batch_size=BATCH_SIZE, text_field=train_dataset.text_field,
                       label_field=train_dataset.label_field, id2label=id2label, verbose=1)
    