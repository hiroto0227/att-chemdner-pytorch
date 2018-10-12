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
    precisions, recalls, fscores = [], [], []
    model.eval()
    eval_iter = Iterator(dataset, batch_size=batch_size, shuffle=False, repeat=False)
    eval_iter.create_batches()

    for batch_i, batch in tqdm(enumerate(eval_iter.batches)):
        tokens = [b.text for b in batch]
        texts = text_field.process([b.text for b in batch], device=-1, train=False)
        labels = label_field.process([b.label for b in batch], device=-1, train=False)
        true_labels = [[id2label[label_id] for label_id in batch] for batch in labels.transpose(1, 0)]
        ###### LSTM CRF ########
        subwords = tokens_batch2subwords(tokens, id2char=id2char, tokenize=lambda x: list(x))
        input_tensor = get_variable(texts, use_gpu=use_gpu)
        subwords_tensor = get_variable(subwords, use_gpu=use_gpu)
        label_ids = model(input_tensor, subwords_tensor)
        pred_labels = [[id2label[int(label_id)] for label_id in batch] for batch in label_ids]

        ###### CHEM以外のラベルを削除。
        true_entities = [true_entity for true_entity in get_entities(true_labels) if true_entity[0] == "CHEM"]
        pred_entities = [pred_entity for pred_entity in get_entities(pred_labels) if pred_entity[0] == "CHEM"]
        
        TP = 0
        pred_num = len(pred_entities)
        true_num = len(true_entities)
        for i, true_entity  in enumerate(true_entities):
            for j, pred_entity in enumerate(pred_entities):
                if true_entity == pred_entity:
                    TP += 1
        p = TP / pred_num if pred_num > 0 else 0
        r = TP / true_num if true_num > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        precisions.append(p)
        recalls.append(r)
        fscores.append(f)
        print("true_num: {}, pred_num: {}, TP: {}, precision: {}, recall: {}, fscore: {}".format(true_num, pred_num, TP, p, r, f))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    fscore = sum(fscores) / len(fscores)

    return precision, recall, fscore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep image inpainting')
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size')
    parser.add_argument('--embed_dim', type=int, default=300, help='embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=1000, help='hidden dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--model_path', type=str, default=None, help='trained model path')
    parser.add_argument('--gpu', action='store_true')
    opt = parser.parse_args()
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2token, id2label = [k for k, v in token2id.items()], [k for k, v in label2id.items()]
    id2char = train_dataset.get_id2token(tokenize=str.split)
    fields = [('text', train_dataset.text_field), ('label', train_dataset.label_field)]
    test_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/test.csv'), fields=fields)
    
    model = LSTMCRFTagger(vocab_dim=len(token2id),
                          tag_dim=len(label2id),
                          batch_size=opt.batch_size,
                          embed_dim=opt.embed_dim,
                          hidden_dim=opt.hidden_dim,
                          num_layers=opt.num_layers,
                          use_gpu=opt.gpu,
                          alphabet_size=len(id2char))

    model.load_state_dict(torch.load(opt.model_path))
    if opt.gpu and torch.cuda.is_available():
        model.cuda()
        print('=========== use gpu ==============')

    p, r, f1 = evaluate(dataset=test_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field,
                       label_field=train_dataset.label_field, id2char=id2char, id2label=id2label, verbose=1, use_gpu=opt.gpu)
    print('\nprecision: {}\nrecall: {}\nf1score: {}'.format(p, r, f1))
