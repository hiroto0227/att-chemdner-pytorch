import os, sys
import torch
import torchtext
from torchtext.data import Iterator
from dataset import ChemdnerDataset
from model.lstm import LSTMTagger
from model.lstm_crf import LSTMCRFTagger
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities
from tqdm import tqdm
from labels import COMMA
import argparse
from utils import get_variable, tokens_batch2subwords


def evaluate(dataset, model, batch_size, text_field, label_field, id2label, id2char, verbose=1, use_gpu=True, use_eval_batch_len=-1):
    all_true_labels = []
    all_pred_labels = []
    model.eval()
    eval_iter = Iterator(dataset, batch_size=batch_size, shuffle=False, repeat=False)
    eval_iter.create_batches()

    for batch_i, batch in tqdm(enumerate(eval_iter.batches)):
        tokens = [b.text for b in batch]
        texts = text_field.process([b.text for b in batch], device=-1, train=False)
        labels = label_field.process([b.label for b in batch], device=-1, train=False)
        true_labels = [[id2label[label_id] for label_id in batch] for batch in labels.transpose(1, 0)]
        ##### LSTM #########
        #output = model(texts)
        #_, label_ids = output.max(2) # (seq_length, batch_size)
        #pred_labels = [[id2label[label_id] for label_id in batch] for batch in label_ids.transpose(1, 0)]
        
        ###### LSTM CRF ########
        subwords = tokens_batch2subwords(tokens, id2char=id2char, tokenize=lambda x: list(x))
        input_tensor = get_variable(texts, use_gpu=use_gpu)
        subwords_tensor = get_variable(subwords, use_gpu=use_gpu)
        label_ids = model(input_tensor, subwords_tensor)

        pred_labels = [[id2label[int(label_id)] for label_id in batch] for batch in label_ids]
        # all_true_labels.extend([t for true_label in true_labels for t in true_label])
        # all_pred_labels.extend([p for pred_label in pred_labels for p in pred_label])

        if verbose:
            for i, (true_label, pred_label) in enumerate(zip(ture_labels, pred_labels)):
                true_entities = get_entities(true_label)
                len_true_entities = len(true_entities)
                pred_entities = get_entities(pred_label)
                len_pred_entities = len(pred_entities)

                if len_true_entities == 0:
                    pass
                else:
                    TP = 0
                    for i, true_entity  in enumerate(true_entities):
                        for j, pred_entity in enumerate(pred_entities):
                            if true_entity == pred_entity:
                                TP += 1
                    precision = TP / len_pred_entities if len_pred_entities > 0 else 0
                    recall = TP / len_true_entities
                    fscore = 2 * precision * recall / (recall + precision) if recall + precision > 0 else 0

        if use_eval_batch_len == batch_i:
            break
    return p, r, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep image inpainting')
    parser.add_argument('--batch_size', type=int, default=10, help='trained batch size')
    parser.add_argument('--model_path', type=str, default=None, help='trained model path')
    parser.add_argument('--gpu', action='store_true')
    opt = parser.parse_args()
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2token, id2label = [k for k, v in token2id.items()], [k for k, v in label2id.items()]
    fields = [('text', train_dataset.text_field), ('label', train_dataset.label_field)]
    test_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/test.csv'), fields=fields)
    
    model = LSTMCRFTagger(len(token2id), len(label2id), batch_size=opt.batch_size, use_gpu=True)
    model.load_state_dict(torch.load(opt.model_path))
    if opt.gpu and torch.cuda.is_available():
        model.cuda()
        print('=========== use gpu ==============')

    p, r, f1 = evaluate(dataset=test_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field,
                       label_field=train_dataset.label_field, id2label=id2label, verbose=0, use_gpu=opt.gpu)
    print('\nprecision: {}\nrecall: {}\nf1score: {}'.format(p, r, f1))
