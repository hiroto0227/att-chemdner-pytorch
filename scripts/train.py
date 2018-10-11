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
from torch.autograd import Variable
from dataset import ChemdnerDataset
from model.lstm_crf import LSTMCRFTagger
from model.attention_lstm import Att_LSTM
from evaluate import evaluate
import pandas as pd
from utils import EarlyStop, get_variable, checkpoint, make_subwords_from_token_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep image inpainting')
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch')
    parser.add_argument('--embed_dim', type=int, default=300, help='embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=1000, help='hidden dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    opt = parser.parse_args()
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '../models/', 'bilstm_crf_{}_{}bs'.format(datetime.now().strftime("%m%d%H%M"), opt.batch_size))
    RESULT_PATH = os.path.join(CURRENT_DIR, '../results/')

    ########### data load #################
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    id2token, id2label = [k for k, v in token2id.items()], [k for k, v in label2id.items()]
    id2char = train_dataset.get_id2token(tokenize=str.split)
    valid_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/valid.csv'))

    model = LSTMCRFTagger(vocab_dim=len(token2id),
                          tag_dim=len(label2id),
                          batch_size=opt.batch_size,
                          embed_dim=opt.embed_dim,
                          hidden_dim=opt.hidden_dim,
                          num_layers=opt.num_layers,
                          use_gpu=opt.gpu,
                          alphabet_size=len(id2char))

    if opt.gpu and torch.cuda.is_available():
        print('\n=============== use GPU =================\n')
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0)

    train_iter = BucketIterator(train_dataset, batch_size=opt.batch_size, shuffle=True, repeat=False, device=-1)
    df_epoch_results = pd.DataFrame(columns=['epoch', 'loss', 'valid_precision', 'valid_recall', 'valid_fscore', 'time'])
    early_stopping = EarlyStop(stop_not_rise_num=7, threshold_rate=0.01)
    precision, recall, f1_score = (0, 0, 0)

    ############ start training ################
    for epoch in range(1, opt.epoch + 1):
        start = time.time()
        loss_per_epoch = 0
        for batch_i, batch in tqdm(enumerate(train_iter)):
            try:
                batch_start = time.time()
                subwords = make_subwords_from_token_batches(batch.text, id2token=id2token, id2char=id2char, tokenize=lambda x: list(x)) # (batch_size, seq_length, subword_length)
                
                subwords_tensor = get_variable(subwords, use_gpu=opt.gpu)
                input_tensor = get_variable(batch.text, use_gpu=opt.gpu)
                target_tensor = get_variable(batch.label, use_gpu=opt.gpu)
                model.zero_grad()
                model.train()
                
                ###### LSTM ##########
                #output = model(input_tensor) # (seq_length, batch_size, tag_size)
                #loss = F.nll_loss(output.view(-1, len(label2id)), output_tensor.view(-1).cpu())
                ###### LStm ##########

                ####### BiLSTM CRF ########
                loss = model.loss(input_tensor, subwords_tensor, target_tensor) / input_tensor.size(0)
                ####### BiLSTM CRF ########
                # print('loss: {}'.format(float(loss)))
                loss.backward()
                optimizer.step()
                loss_per_epoch += float(loss)

            except:
                checkpoint(epoch, model, MODEL_PATH, opt.batch_size, interrupted=True, use_gpu=opt.gpu)
                df_epoch_results.to_csv(os.path.join(RESULT_PATH, 'result_epoch_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')
                traceback.print_exc()
                sys.exit(1)

        if epoch % 5 == 0:
            precision, recall, f1_score = evaluate(dataset=valid_dataset, 
                                                   model=model,
                                                   batch_size=opt.batch_size,
                                                   text_field=train_dataset.text_field,
                                                   label_field=train_dataset.label_field,
                                                   id2label=id2label,
                                                   id2char=id2char,
                                                   verbose=0,
                                                   use_gpu=opt.gpu)
            if early_stopping.is_end(f1_score):
                break

        df_epoch_results = df_epoch_results.append(pd.Series({'epoch': epoch,
                           'loss': loss_per_epoch,
                           'valid_precision': precision,
                           'valid_recall': recall,
                           'valid_fscore': f1_score,
                           'time': time.time() - start}), ignore_index=True)

        print('{}epoch\nloss: {}\nvalid: {}\ntime: {} sec.\n'.format(epoch, loss_per_epoch, f1_score, time.time() - start))
    
    checkpoint(epoch, model, MODEL_PATH, batch_size=opt.batch_size, use_gpu=opt.gpu)
    df_epoch_results.to_csv(os.path.join(RESULT_PATH, 'result_epoch_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')
