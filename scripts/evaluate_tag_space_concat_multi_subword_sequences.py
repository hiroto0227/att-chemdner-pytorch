import re
import os
import torch
from torchtext.data import Iterator
from dataset import ChemdnerDataset
from seqeval.metrics.sequence_labeling import get_entities
from tqdm import tqdm
from model.multi_subword_attention import MultiSubwordAttentionTagger
from data.processed import tokenize
from utils import get_variable


def evaluate(dataset, model, batch_size, train_fields, subword_tokenizers, verbose=1, use_gpu=True):
    precisions, recalls, fscores = [], [], []
    model.eval()
    eval_iter = Iterator(dataset, batch_size=batch_size, shuffle=False, repeat=False)
    eval_iter.create_batches()
    id2label = train_fields["label"].vocab.itos

    TP = 0
    pred_num = 0
    true_num = 0
    for batch_i, batch in tqdm(enumerate(eval_iter.batches)):
        label_sequences = train_fields["label"].process([b.label for b in batch], device=-1, train=False)
        true_labels = [[id2label[label_id] for label_id in batch] for batch in label_sequences.transpose(1, 0)]
        x_char = get_variable(train_fields["char"].process([b.char for b in batch], device=-1, train=False), use_gpu=use_gpu)
        x_subs = []
        for name in subword_tokenizers.keys():
           x_subs.append(get_variable(train_fields[name].process([b.__getattribute__(name) for b in batch], device=-1, train=False), use_gpu=use_gpu))
        label_ids = model(x_char, x_subs)
        pred_labels = [[id2label[int(label_id)] for label_id in batch] for batch in label_ids]

        ###### CHEM以外のラベルを削除。
        true_entities = [true_entity for true_entity in get_entities(true_labels) if true_entity[0] == "CHEM"]
        pred_entities = [pred_entity for pred_entity in get_entities(pred_labels) if pred_entity[0] == "CHEM"]

        pred_num += len(pred_entities)
        true_num += len(true_entities)
        for i, true_entity in enumerate(true_entities):
            for j, pred_entity in enumerate(pred_entities):
                if true_entity == pred_entity:
                    TP += 1
    precision = TP / pred_num if pred_num > 0 else 0
    recall = TP / true_num if true_num > 0 else 0
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("true_num: {}, pred_num: {}, TP: {}, precision: {}, recall: {}, fscore: {}".format(true_num, pred_num, TP, precision, recall, fscore))
    return precision, recall, fscore

