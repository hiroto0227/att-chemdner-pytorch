import os
import json
import torch
import argparse
from tqdm import tqdm
import dataset
import labels
from model.bilstmlstmcrf import BiLSTMLSTMCRF
from utils import get_variable
from sentencepieces.sp_tokenizer import SentencePieceTokenizer
from chemdnerdatautils import file2sequences, labels_to_anns, annotations_to_spantokens, text_to_spantokens


def evaluate(train_path, test_path, model_path, config_path, verbose=1):
    precisions, recalls, fscores = [], [], []
    TP = 0
    pred_num = 0
    true_num = 0
    with open(config_path, "rt") as f:
        opt = json.load(f)
    
    ######### load train for vocabulary #########
    sp = SentencePieceTokenizer()
    sp.load(opt["sp_model"])
    train_token_seqs, _ = dataset.load_sequences(train_path, sp.tokenize)
    token2id, char2id = dataset.make_vocab(train_token_seqs)
    label2id = {labels.O: 0, labels.B: 1, labels.I: 2}
    id2label = [key for key in label2id.keys()]
    del train_token_seqs, _

    ######### laod pretrain embedding for OOV ########
    pretrain_embed = None

    ######### load model ###########
    model = BiLSTMLSTMCRF(word_vocab_dim=len(token2id),
                          char_vocab_dim=len(char2id),
                          label_dim=len(label2id),
                          word_embed_dim=opt["word_embed"],
                          char_embed_dim=opt["char_embed"],
                          word_lstm_dim=opt["word_lstm"],
                          char_lstm_dim=opt["char_lstm"],
                          batch_size=1,
                          pretrain_embed=pretrain_embed,
                          training=False,
                          use_gpu=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    fileids = [filename.replace('.txt', '') for filename in os.listdir(test_path) if filename.endswith('.txt')]
    for i, fileid in tqdm(enumerate(fileids)):
        with open(os.path.join(test_path, fileid + '.txt'), 'rt') as f:
            text = f.read()
        with open(os.path.join(test_path, fileid + '.ann'), 'rt') as f:
            annotations = f.read().split('\n')
        text_spantokens = text_to_spantokens(text, sp.tokenize)
        true_ann_spantokens = annotations_to_spantokens(annotations)

        token_seq, label_seq = file2sequences(test_path, fileid, sp.tokenize)
        token_id_seqs = torch.LongTensor(dataset.to_id([token_seq], token2id)).transpose(1, 0)
        char_seq = [[c for c in token] for token in token_seq]
        char_id_seqs = dataset.to_id([char_seq], char2id, char=True)
        char_id_seqs = dataset.padding(char_id_seqs, len(char_id_seqs[0]), char2id[labels.PAD], char_max_len=max([len(c) for c in char_id_seqs[0]]))
        char_id_seqs = torch.LongTensor(char_id_seqs).transpose(1, 0)
        label_id_seqs = dataset.to_id([label_seq], label2id, label=True)
        pred_label_ids = model(token_id_seqs, char_id_seqs)
        pred_labels = [id2label[int(label_id)] for label_id in pred_label_ids.squeeze(0)]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate chemdner corpus")
    parser.add_argument("--test-path", type=str, help="test path")
    parser.add_argument("--train-path", type=str, help="train path (for token2id)")
    parser.add_argument("--model-path", type=str, help="model path")
    parser.add_argument("--config-path", type=str, help="train config path")
    opt = parser.parse_args()

    evaluate(opt.train_path, opt.test_path, opt.model_path, opt.config_path)
