import argparse
import os, sys
import traceback
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
import torch
import torchtext
from torchtext.data import BucketIterator, Iterator
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from dataset import ChemdnerDataset
from model.lstm import LSTMTagger
from model.attention_lstm import Att_LSTM
from evaluate import evaluate


def checkpoint(epoch, model, model_path):
    print('model saved!!')
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
    id2label = [k for k, v in label2id.items()]
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
                # print('\ninput: {}'.format(batch.text.shape)) # (seq_length, batch_size)
                output = model(batch.text) # (seq_length, batch_size, tag_size)
                # print('output: {}'.format(output.shape))
                loss = F.nll_loss(output.view(-1, len(label2id)), batch.label.view(-1))
                loss.backward()
                optimizer.step()
                print('loss: {}'.format(loss))
                loss_per_epoch += float(loss)
            except:
                checkpoint(epoch, model, MODEL_PATH)
                traceback.print_exc()
                sys.exit(1)
            #if batch_i % 10 == 0:
            #    print('--- evaluate ---')
            #    valid_f1_score = evaluate(dataset=valid_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field,
            #                      label_field=train_dataset.label_field, id2label=id2label, verbose=0)
            #    print('valid: {}'.format(valid_f1_score))
        valid_f1_score = evaluate(dataset=valid_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field,
                                  label_field=train_dataset.label_field, id2label=id2label, verbose=0)
        print('{}epoch\nloss: {}\nvalid: {}\ntime: {} sec.\n'.format(epoch, loss_per_epoch, valid_f1_score, time.time() - start))
        checkpoint(epoch, model, MODEL_PATH)

