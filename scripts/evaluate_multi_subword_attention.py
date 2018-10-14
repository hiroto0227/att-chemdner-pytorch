import re
import os
import torch
from torchtext.data import Iterator
from dataset import ChemdnerDataset
from seqeval.metrics.sequence_labeling import get_entities
from tqdm import tqdm
import argparse
from model.multi_subword_attention import MultiSubwordAttentionTagger
from data.processed import tokenize
from utils import get_variable


def evaluate(dataset, model, batch_size, train_fields, verbose=1, use_gpu=True, use_eval_batch_len=-1):
    precisions, recalls, fscores = [], [], []
    model.eval()
    eval_iter = Iterator(dataset, batch_size=batch_size, shuffle=False, repeat=False)
    eval_iter.create_batches()

    for batch_i, batch in tqdm(enumerate(eval_iter.batches)):
        label_sequences = train_dataset.fields["label"].process([b.label for b in batch], device=-1, train=False)
        true_labels = [[id2label[label_id] for label_id in batch] for batch in label_sequences.transpose(1, 0)]
        x_char = get_variable(train_dataset.fields["char"].process([b.char for b in batch], device=-1, train=False))
        x_subs = []
        for name in subword_tokenizers.keys():
            for b in batch:
                x_subs.append(get_variable(train_dataset.fields[name].process(b.__getattribute__(name), device=-1, train=False)))
        label_ids = model(x_char, x_subs)
        pred_labels = [[id2label[int(label_id)] for label_id in batch] for batch in label_ids]

        ###### CHEM以外のラベルを削除。
        true_entities = [true_entity for true_entity in get_entities(true_labels) if true_entity[0] == "CHEM"]
        pred_entities = [pred_entity for pred_entity in get_entities(pred_labels) if pred_entity[0] == "CHEM"]

        TP = 0
        pred_num = len(pred_entities)
        true_num = len(true_entities)
        for i, true_entity in enumerate(true_entities):
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


def sub1_tokenize(x):
    return re.split('( | |\xa0|\t|\n|[0-9])', x)


def sub2_tokenize(x):
    return re.split('( | |\xa0|\t|\n|[0-9]|[A-Z])', x)

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

    subword_tokenizers = {
        "sub1": sub1_tokenize,
        "sub2": sub2_tokenize,
        "sub3": tokenize
    }

    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2token, id2label = [k for k, v in token2id.items()], [k for k, v in label2id.items()]
    id2char = train_dataset.get_id2token(tokenize=str.split)
    fields = [('text', train_dataset.text_field), ('label', train_dataset.label_field)]
    test_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/test.csv'), fields=fields)

    model = MultiSubwordAttentionTagger(char_vocab_dim=len(train_dataset.fields["char"].vocab.itos),
                                        sub_vocab_dims=[len(train_dataset.fields[name].vocab.itos) for name in subword_tokenizers.keys()],
                                        tag_dim=len(train_dataset.fields["label"].vocab.itos),
                                        subwords_num=len(subword_tokenizers),
                                        char_embed_dim=50,
                                        sub_embed_dims=[100 for i in len(subword_tokenizers)],
                                        batch_size=opt.batch_size,
                                        hidden_dim=opt.hidden_dim,
                                        use_gpu=opt.gpu)
    model.load_state_dict(torch.load(opt.model_path))
    if opt.gpu and torch.cuda.is_available():
        model.cuda()
        print('=========== use gpu ==============')

    p, r, f1 = evaluate(dataset=test_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field,
                        label_field=train_dataset.label_field, id2char=id2char, id2label=id2label, verbose=1, use_gpu=opt.gpu)
    print('\nprecision: {}\nrecall: {}\nf1score: {}'.format(p, r, f1))
