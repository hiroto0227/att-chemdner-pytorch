import argparse
import json
import torch
import os
from tqdm import tqdm
import time
from torch import optim
import dataset
from model.bilstmlstmcrf import BiLSTMLSTMCRF
import pandas as pd
from chemdnerdatautils import tokenize
from utils import checkpoint, get_variable
from pretrain import pretrain
import labels
from sentencepieces.sp_tokenizer import SentencePieceTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep image inpainting')
    parser.add_argument('--batch-size', '-bs', type=int, default=100, help='training batch size')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch')
    parser.add_argument('--word-embed', type=int, default=50)
    parser.add_argument('--char-embed', type=int, default=30)
    parser.add_argument('--word-lstm', type=int, default=200)
    parser.add_argument('--char-lstm', type=int, default=50)
    parser.add_argument('--sp-model', type=str, default=None, help='sentencepiece model path')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--weight-decay', type=float, default=1e-8)
    parser.add_argument('--word2vec-path', type=str, default=None, help='pretrained word2vec path')
    parser.add_argument('--model-path', type=str, default='./chemdner.pth', help='path to save model')
    parser.add_argument('--gpu', action='store_true')
    opt = parser.parse_args()
    with open(opt.model_path + ".param.txt", "wt") as f:
        json.dump(vars(opt), f)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    sp = SentencePieceTokenizer()
    sp.load(opt.sp_model)
    tokenize = sp.tokenize

    ########### data load #################
    token_seqs, label_seqs = dataset.load_sequences(os.path.join(CURRENT_DIR, '../datas/raw/train'), tokenize)
    char_seqs = [[[c for c in token] for token in token_seq] for token_seq in token_seqs]
    token2id, char2id = dataset.make_vocab(token_seqs)
    label2id = {labels.O: 0, labels.B: 1, labels.I: 2}
    token_id_seqs = dataset.to_id(token_seqs, token2id)
    char_id_seqs = dataset.to_id(char_seqs, char2id, char=True)
    label_id_seqs = dataset.to_id(label_seqs, label2id, label=True)

    ######### load pretrain embedding ################
    word2vec = pretrain.load_word2vec(opt.word2vec_path)
    pretrain_embed = pretrain.make_pretrain_embed(word2vec, token2id, opt.word_embed)

    ########## model init ##################
    model = BiLSTMLSTMCRF(word_vocab_dim=len(token2id),
                          char_vocab_dim=len(char2id),
                          label_dim=len(label2id),
                          word_embed_dim=opt.word_embed,
                          char_embed_dim=opt.char_embed,
                          word_lstm_dim=opt.word_lstm,
                          char_lstm_dim=opt.char_lstm,
                          batch_size=opt.batch_size,
                          pretrain_embed=pretrain_embed,
                          training=True,
                          use_gpu=opt.gpu)
    print("\n", model)
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    ############# train parameter init #####################
    df_epoch_results = pd.DataFrame(columns=['epoch', 'loss', 'valid_precision', 'valid_recall', 'valid_fscore', 'time'])
    precision, recall, f1_score = (0, 0, 0)

    ############ start training ################
    for epoch in range(1, opt.epoch + 1):
        start = time.time()
        loss_per_epoch = 0
        for i, (token_batch, char_batch, label_batch) in tqdm(enumerate(dataset.batch_gen(token_id_seqs, char_id_seqs, label_id_seqs, opt.batch_size, token2id[labels.PAD], char2id[labels.PAD], label2id[labels.O], shuffle=True))):
            batch_start = time.time()
            model.zero_grad()
            model.train()
            token_batch = get_variable(torch.LongTensor(token_batch), use_gpu=opt.gpu).transpose(1, 0)
            char_batch = get_variable(torch.LongTensor(char_batch), use_gpu=opt.gpu).transpose(1, 0)
            label_batch = get_variable(torch.LongTensor(label_batch), use_gpu=opt.gpu).transpose(1, 0)
            # loss = model.loss(token_batch, char_batch, label_batch) / token_batch.shape[0]
            loss = model.loss(token_batch, char_batch, label_batch)
            optimizer.zero_grad()
            print("loss: {}".format(loss))
            loss.backward()
            optimizer.step()
            #print("loss: {}".format(float(loss)))
            loss_per_epoch += float(loss)
        
        print('{}epoch\nloss: {}\nvalid: {}\ntime: {} sec.\n'.format(epoch, loss_per_epoch, 0, time.time() - start))
        if epoch % 10 == 0:
            print("model save {}epoch".format(epoch))
            checkpoint(model, opt.model_path)
    checkpoint(model, opt.model_path)
