import os, sys
import torch
import torchtext
from dataset import ChemdnerDataset
from model.lstm import LSTMTagger
from seqeval.metrics.sequence_labeling import get_entities
from data.processed import tokenize, labels_to_spantokens, text_to_spantokens

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '../models/lstm_one_epoch_64batch')
    
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2label = [v for k, v in label2id.items()]
    fields = [('text', train_dataset.text_field), ('label', train_dataset.label_field)]
    
    model = LSTMTagger(len(token2id), len(label2id), batch_size=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    text = input('>>')
    inp = torch.LongTensor([[token2id[token]] for token in tokenize(text)])
    out = model(inp)
    _, label_ids = out.max(1)
    pred_labels = [id2label[label_id] for label_id in label_ids[0]]
    spantokens = labels_to_spantokens(pred_labels, text_to_spantokens(text))
    print(spantokens)
