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
import pandas as pd


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
    RESULT_PATH = os.path.join(CURRENT_DIR, '../results/')
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2label = [k for k, v in label2id.items()]
    valid_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/test.csv'))
    
    model = LSTMTagger(vocab_dim=len(token2id), tag_dim=len(label2id), batch_size=opt.batch_size)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_sum = 0
    train_iter = BucketIterator(train_dataset, batch_size=opt.batch_size, shuffle=True, repeat=False)
    
    df_epoch_results = pd.DataFrame(columns=['epoch', 'loss', 'valid_precision', 'valid_recall', 'valid_fscore', 'time'])
    df_iteration_results = pd.DataFrame(columns=['iteration', 'loss', 'time'])

    for epoch in range(opt.epoch):
        start = time.time()
        loss_per_epoch = 0
        for batch_i, batch in tqdm(enumerate(train_iter)):
            try:
                batch_start = time.time()
                model.zero_grad()
                model.train()
                # print('\ninput: {}'.format(batch.text.shape)) # (seq_length, batch_size)
                output = model(batch.text.cpu()) # (seq_length, batch_size, tag_size)
                # print('output: {}'.format(output.shape))
                loss = F.nll_loss(output.view(-1, len(label2id)), batch.label.view(-1).cpu())
                loss.backward()
                optimizer.step()
                print('loss: {}'.format(loss))
                loss_per_epoch += float(loss)
                df_iteration_results = df_iteration_results.append(pd.Series({'iteration': '{}ep_{}iter'.format(epoch + 1, batch_i), 'loss': float(loss), 'time': time.time() - batch_start}), ignore_index=True)
            except:
                checkpoint(epoch, model, MODEL_PATH, interrupted=True)
                traceback.print_exc()
                sys.exit(1)
        precision, recall, f1_score = evaluate(dataset=valid_dataset, model=model, batch_size=opt.batch_size, text_field=train_dataset.text_field, label_field=train_dataset.label_field, id2label=id2label, verbose=0)
        print('{}epoch\nloss: {}\nvalid: {}\ntime: {} sec.\n'.format(epoch + 1, loss_per_epoch, f1_score, time.time() - start))
        df_epoch_results = df_epoch_results.append(pd.Series({'epoch': epoch + 1, 'loss': loss_per_epoch, 'valid_precision': precision, 'valid_recall': recall, 'valid_fscore': f1_score, 'time': time.time() - start}), ignore_index=True)
        checkpoint(epoch, model, MODEL_PATH)

    df_epoch_results.to_csv(os.path.join(RESULT_PATH, 'result_epoch_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')
    df_iteration_results.to_csv(os.path.join(RESULT_PATH, 'result_iter_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')
