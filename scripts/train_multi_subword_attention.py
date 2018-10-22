import re
import argparse
import os
import sys
import traceback
from datetime import datetime
from tqdm import tqdm
import time
import torch
from torchtext.data import BucketIterator
from torch import optim
from dataset import ChemdnerSubwordDataset
from model.multi_subword_attention import MultiSubwordAttentionTagger
from evaluate_multi_subword_attention import evaluate
import pandas as pd
from data.processed import tokenize
from utils import EarlyStop, get_variable, checkpoint
import gc
from itertools import chain
from collections import deque
from memory_profiler import profile

def total_size(obj, verbose=False):
    seen = set()
    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default=0)
        if verbose:
            print(s, type(o), repr(o))
        if isinstance(o, (tuple, list, set, frozenset, deque)):
            s += sum(map(sizeof, iter(o)))
        elif isinstance(o, dict):
            s += sum(map(sizeof, chain.from_iterable(o.items())))
        elif "__dict__" in dir(o):  # もっと良い方法はあるはず
            s += sum(map(sizeof, chain.from_iterable(o.__dict__.items())))
        return s

    return sizeof(obj)


def sub1_tokenize(x):
    return re.split('( | |\xa0|\t|\n|[0-9])', x)


def sub2_tokenize(x):
    return re.split('( | |\xa0|\t|\n|[0-9]|[A-Z])', x)

@profile
def train():
    parser = argparse.ArgumentParser(description='deep image inpainting')
    parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch')
    parser.add_argument('--embed_dim', type=int, default=100, help='embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=50, help='hidden dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    opt = parser.parse_args()

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '../models/', 'multi_sub_lstm_{}_{}bs'.format(datetime.now().strftime("%Y%m%d%H%M"), opt.batch_size))
    RESULT_PATH = os.path.join(CURRENT_DIR, '../results/')

    ########### data load #################
    subword_tokenizers = {
        "sub1": tokenize
        #"sub2": sub2_tokenize,
        #"sub3": tokenize
    }
    train_dataset = ChemdnerSubwordDataset(path=os.path.join(CURRENT_DIR, '../datas/raw/train'), subword_tokenizers=subword_tokenizers)
    train_dataset.make_vocab()
    train_fields = train_dataset.fields

    model = MultiSubwordAttentionTagger(char_vocab_dim=len(train_dataset.fields["char"].vocab.itos),
                                        sub_vocab_dims=[len(train_dataset.fields[name].vocab.itos) for name in subword_tokenizers.keys()],
                                        tag_dim=len(train_dataset.fields["label"].vocab.itos),
                                        char_embed_dim=25,
                                        sub_embed_dims=[50 for i in range(len(subword_tokenizers))],
                                        batch_size=opt.batch_size,
                                        hidden_dim=opt.hidden_dim,
                                        use_gpu=opt.gpu)
    print("\nmodel: {}\n".format(model))
    if opt.gpu and torch.cuda.is_available():
        print('\n=============== use GPU =================\n')
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0)

    train_iter = BucketIterator(train_dataset, batch_size=opt.batch_size, shuffle=True, repeat=False, device=-1)
    df_epoch_results = pd.DataFrame(columns=['epoch', 'loss', 'valid_precision', 'valid_recall', 'valid_fscore', 'time'])
    early_stopping = EarlyStop(stop_not_rise_num=7, threshold_rate=0.01)
    precision, recall, f1_score = (0, 0, 0)
    del train_dataset

    ############ start training ################
    for epoch in range(1, opt.epoch + 1):
        start = time.time()
        loss_per_epoch = 0
        for batch_i, batch in tqdm(enumerate(train_iter)):
            if epoch == 1 and batch_i == 0:
                del batch.dataset
            gc.collect()
            try:
                optimizer.zero_grad()
                batch_start = time.time()
                x_char = get_variable(batch.char, use_gpu=opt.gpu)
                x_subs = [get_variable(batch.__getattribute__(name), use_gpu=opt.gpu) for name in subword_tokenizers.keys()]
                y = get_variable(batch.label, use_gpu=opt.gpu)
                #print("\nbatch: {}".format(total_size(batch)))
                #print("model: {}".format(total_size(model)))
                del batch
                model.zero_grad()
                model.train()
                loss = model.loss(x_char, x_subs, y) / x_char.size(0)
                # print('loss: {}'.format(float(loss)))
                loss.backward()
                optimizer.step()
                loss_per_epoch += float(loss)
            except:
                checkpoint(epoch, model, MODEL_PATH, opt.batch_size, interrupted=True, use_gpu=opt.gpu)
                df_epoch_results.to_csv(os.path.join(RESULT_PATH, 'result_epoch_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')
                traceback.print_exc()
                sys.exit(1)
        if epoch % 5 == 1:
            valid_dataset = ChemdnerSubwordDataset(path=os.path.join(CURRENT_DIR, '../datas/raw/valid'), subword_tokenizers=subword_tokenizers)
            precision, recall, f1_score = evaluate(dataset=valid_dataset,
                                                   model=model,
                                                   batch_size=opt.batch_size,
                                                   train_fields=train_fields,
                                                   subword_tokenizers=subword_tokenizers,
                                                   verbose=0,
                                                   use_gpu=opt.gpu)
            if early_stopping.is_end(f1_score):
                break
            del valid_dataset

        df_epoch_results = df_epoch_results.append(pd.Series({'epoch': epoch,
                                                              'loss': loss_per_epoch,
                                                              'valid_precision': precision,
                                                              'valid_recall': recall,
                                                              'valid_fscore': f1_score,
                                                              'time': time.time() - start}), ignore_index=True)

        print('{}epoch\nloss: {}\nvalid: {}\ntime: {} sec.\n'.format(epoch, loss_per_epoch, f1_score, time.time() - start))
    checkpoint(epoch, model, MODEL_PATH, batch_size=opt.batch_size, use_gpu=opt.gpu)
    df_epoch_results.to_csv(os.path.join(RESULT_PATH, 'result_epoch_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')

if __name__ == "__main__":
    train()
