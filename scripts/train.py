import argparse
import os, sys
import traceback
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import torchtext
from torchtext.data import BucketIterator, Iterator
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from dataset import ChemdnerDataset
from model.lstm import LSTMTagger
from model.attention_lstm import Att_LSTM
from evaluate import evaluate
from labels import UNK, PAD


def make_subwords_from_token_batches(token_batch, id2token, subword2id, tokenize, batch_first=True):
    """入力のtokenをsubword化して返す。
    input:token_batch
        size=(batch_size, seq_len, 1)
    output:subwords
        size=(batch_size, seq_len, max_subword_len)
    """
    token_ids_batch = token_batch #(batch_size, seq_len, 1)
    subword_batch = []
    max_subword_length = 0
    for token_ids in token_ids_batch:
        tokens = [id2token[token_id] for token_id in token_ids] #(seq_len)
        subwords = []
        for i, token in enumerate(tokens):
            if not token in [UNK, PAD, '<pad>']:
                subwords.append(np.array([subword2id[subword] for subword in tokenize(token)]))
            else:
                subwords.append(np.array([token2id[token]]))
            max_subword_length = len(subwords[-1]) if len(subwords[-1]) > max_subword_length else max_subword_length
        subword_batch.append(subwords) # (batch_size, seq_len, subword_size)
    padded_subwords = [[np.pad(subwords, mode='constant', pad_width=(0, max_subword_length - len(subwords)))
                        for subwords in batch] for batch in subword_batch]
    return torch.LongTensor(padded_subwords) # (batch_size, seq_len, subword_size)    


def checkpoint(epoch, model, model_path, interrupted=False):
    print('model saved!!')
    if interrupted:
        torch.save(model.state_dict(), MODEL_PATH + '_{}ep_{}bs_interrupted.pth'.format(epoch, opt.batch_size))
    else:
        torch.save(model.state_dict(), MODEL_PATH + '_{}ep_{}bs.pth'.format(epoch, opt.batch_size))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep image inpainting')
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch')
    opt = parser.parse_args()

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '../models/', 'lstm_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2token, id2label = [k for k, v in token2id.items()], [k for k, v in label2id.items()]
    valid_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/test.csv'))
    
    model = LSTMTagger(vocab_dim=len(token2id), tag_dim=len(label2id), batch_size=opt.batch_size)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_sum = 0
    train_iter = BucketIterator(train_dataset, batch_size=opt.batch_size, shuffle=True, repeat=False)

    for epoch in range(opt.epoch):
        start = time.time()
        loss_per_epoch = 0
        for batch_i, batch in tqdm(enumerate(train_iter)):
            try:
                model.zero_grad()
                model.train()
                subwords = make_subwords_from_token_batches(batch.text, id2token=id2token, subword2id=token2id,
                                                            tokenize=lambda x: list(x)) # (batch_size, seq_length, subword_length)
                print(subwords.shape)
                del subwords
                # print('\ninput: {}'.format(batch.text.shape)) # (batch_size, seq_length)
                output = model(batch.text) # (batch_size, seq_length, tag_size)
                # print('output: {}'.format(output.shape))
                loss = F.nll_loss(output.view(-1, len(label2id)), batch.label.view(-1))
                loss.backward()
                optimizer.step()
                print('loss: {}'.format(loss))
                loss_per_epoch += float(loss)
            except:
                checkpoint(epoch, model, MODEL_PATH, interrupted=True)
                traceback.print_exc()
                sys.exit(1)
            if batch_i % 10 == 0:
                print('--- evaluate ---')
                valid_f1_score = evaluate(dataset=valid_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field,
                                  label_field=train_dataset.label_field, id2label=id2label, verbose=0)
                print('valid: {}'.format(valid_f1_score))
        valid_f1_score = evaluate(dataset=valid_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field,
                                  label_field=train_dataset.label_field, id2label=id2label, verbose=0)
        print('{}epoch\nloss: {}\nvalid: {}\ntime: {} sec.\n'.format(epoch, loss_per_epoch, valid_f1_score, time.time() - start))
        checkpoint(epoch, model, MODEL_PATH)

