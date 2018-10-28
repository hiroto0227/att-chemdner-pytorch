import os
import torch
from tqdm import tqdm
from utils import get_variable
from datautils import labels_to_anns, text_to_spantokens, annotations_to_spantokens


def evaluate(eval_data_path, model, batch_size, token_field, label_field, tokenizer, verbose=1, use_gpu=True):
    precisions, recalls, fscores = [], [], []
    model.eval()
    id2label = label_field.vocab.itos

    TP = 0
    pred_num = 0
    true_num = 0
    fileids = [filename.replace('.txt', '') for filename in os.listdir(eval_data_path) if filename.endswith('.txt')]
    for i, fileid in tqdm(enumerate(fileids)):
        #if i == 50:
        #    break
        with open(os.path.join(eval_data_path, fileid + '.txt'), 'rt') as f:
            text = f.read()
        with open(os.path.join(eval_data_path, fileid + '.ann'), 'rt') as f:
            annotations = f.read().split('\n')
        text_spantokens = text_to_spantokens(text, tokenizer)
        true_ann_spantokens = annotations_to_spantokens(annotations)
        x = get_variable(token_field.process([tokenizer(text)], device=-1, train=False), use_gpu=use_gpu)
        label_ids = model(x)
        pred_labels = [id2label[int(label_id)] for label_id in label_ids.squeeze(0)]
        pred_ann_spantokens = labels_to_anns(pred_labels, text_spantokens)
        pred_num += len(pred_ann_spantokens)
        true_num += len(true_ann_spantokens)
        TP += len(set(pred_ann_spantokens) & set(true_ann_spantokens))
        if verbose:
            print('===========================')
            print(true_ann_spantokens)
            print('-----------------')
            print(pred_ann_spantokens)
            print('-----------------')
            print(set(pred_ann_spantokens) & set(true_ann_spantokens))
            print('===========================')
    precision = TP / pred_num if pred_num > 0 else 0
    recall = TP / true_num if true_num > 0 else 0
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("true_num: {}, pred_num: {}, TP: {}, precision: {}, recall: {}, fscore: {}".format(true_num, pred_num, TP, precision, recall, fscore))
    return precision, recall, fscore
