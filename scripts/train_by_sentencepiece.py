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
from dataset import TokenizeDataset
from model.bilstmcrf import BiLSTMCRF
from model.word2vec import Word2Vec
from evaluate_by_tokenizer import evaluate
import pandas as pd
import numpy as np
from utils import EarlyStop, get_variable, checkpoint
import torch.nn as nn
from sentencepieces.sp_tokenizer import SentencePieceTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep image inpainting')
    parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch')
    parser.add_argument('--lm-epoch', type=int, default=1, help='training epoch')
    parser.add_argument('--embed_dim', type=int, default=100, help='embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=400, help='hidden dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--use-pretrain', type=str, default='', help='num layers')
    parser.add_argument('--sp-model', type=str, default='', help='num layers')
    parser.add_argument('--gpu', action='store_true')
    opt = parser.parse_args()

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '../models/', 'tokenize_char_bilstmcrf_{}_{}bs'.format(datetime.now().strftime("%Y%m%d%H%M"), opt.batch_size))
    RESULT_PATH = os.path.join(CURRENT_DIR, '../results/')

    ########### data load #################
    sp = SentencePieceTokenizer()
    sp.load(opt.sp_model)
    tokenizer = sp.tokenize
    train_dataset = TokenizeDataset(path=os.path.join(CURRENT_DIR, '../datas/raw/train'), tokenizer=tokenizer)
    train_dataset.token_field.build_vocab(train_dataset)
    train_dataset.label_field.build_vocab(train_dataset)
    print(train_dataset.token_field.vocab.itos)
    train_iter = BucketIterator(train_dataset, batch_size=opt.batch_size, shuffle=True, repeat=False, device=-1)
    ########## model init ##################
    lm_model = Word2Vec(vocab_dim=len(train_dataset.token_field.vocab.itos),
                                 batch_size=opt.batch_size,
                                 embed_dim=opt.embed_dim,
                                 hidden_dim=opt.hidden_dim,
                                 use_gpu=opt.gpu)
    model = BiLSTMCRF(vocab_dim=len(train_dataset.token_field.vocab.itos),
                      tag_dim=len(train_dataset.label_field.vocab.itos),
                      batch_size=opt.batch_size,
                      embed_dim=opt.embed_dim,
                      hidden_dim=opt.hidden_dim,
                      num_layers=opt.num_layers,
                      use_gpu=opt.gpu)

    criterion_lm = nn.MSELoss()
    optimizer_lm = optim.SGD(lm_model.parameters(), lr=0.01, weight_decay=0)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)

    if opt.gpu and torch.cuda.is_available():
        print('\n=============== use GPU =================\n')
        lm_model.cuda()
        model.cuda()
        criterion_lm.cuda()

    ############# train parameter init #####################
    df_epoch_results = pd.DataFrame(columns=['epoch', 'loss', 'valid_precision', 'valid_recall', 'valid_fscore', 'time'])
    precision, recall, f1_score = (0, 0, 0)

    ############ pretrain ###############
    for epoch in range(1, opt.lm_epoch + 1):
        forward_losses = 0
        backward_losses = 0
        for batch_i, batch in tqdm(enumerate(train_iter)):
            x = get_variable(batch.token, use_gpu=opt.gpu)
            reverse_x = torch.from_numpy(np.flip(batch.token.numpy(), axis=0).copy())
            # forward lm
            next_x = batch.token[1:, :]
            pad = torch.full((1, next_x.shape[1]), train_dataset.token_field.vocab.itos.index('<pad>'), dtype=torch.long)
            next_x = torch.cat((next_x, pad))
            output = lm_model(get_variable(batch.token, use_gpu=opt.gpu))
            target = lm_model.embedding(get_variable(next_x, use_gpu=opt.gpu))
            target.detach_()
            target = target.view(-1, opt.embed_dim).float()
            output = output.view(-1, opt.embed_dim).float()
            loss = criterion_lm(output, target) / batch.token.size(0)
            loss.backward()
            forward_losses += float(loss)
            optimizer_lm.step()

        print("\nforward LM loss: {}".format(forward_losses))
        print("backward LM loss: {}".format(backward_losses))

    ############ transport pretrained layers ###############
    for key, state in lm_model.state_dict().items():
        model.state_dict()[key] = state
    checkpoint(opt.lm_epoch, model, os.path.join(CURRENT_DIR, '../models/pretrained_bilstm_crf.pth'), opt.batch_size, interrupted=False, use_gpu=opt.gpu)
    
    if opt.use_pretrain:
        print("========== use pretrain model ===========")
        model.load_state_dict(torch.load(opt.use_pretrain))

    print(model)

    ############ start training ################
    train_iter = BucketIterator(train_dataset, batch_size=opt.batch_size, shuffle=True, repeat=False, device=-1)
    for epoch in range(1, opt.epoch + 1):
        start = time.time()
        loss_per_epoch = 0
        for batch_i, batch in tqdm(enumerate(train_iter)):
            try:
                batch_start = time.time()
                x = get_variable(batch.token, use_gpu=opt.gpu)
                y = get_variable(batch.label, use_gpu=opt.gpu)
                model.zero_grad()
                model.train()
                loss = model.loss(x, y) / x.size(0)
                #print('NER loss: {}'.format(float(loss)))
                loss.backward()
                optimizer.step()
                loss_per_epoch += float(loss)
            except:
                checkpoint(epoch, model, MODEL_PATH, opt.batch_size, interrupted=True, use_gpu=opt.gpu)
                df_epoch_results.to_csv(os.path.join(RESULT_PATH, 'result_epoch_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')
                traceback.print_exc()
                sys.exit(1)
        if epoch % 5 == 1:
            precision, recall, f1_score = evaluate(eval_data_path=os.path.join(CURRENT_DIR, '../datas/raw/valid'),
                                                   model=model,
                                                   batch_size=opt.batch_size,
                                                   token_field=train_dataset.token_field,
                                                   label_field=train_dataset.label_field,
                                                   tokenizer=tokenizer,
                                                   verbose=0,
                                                   use_gpu=opt.gpu)
            df_epoch_results = df_epoch_results.append(pd.Series({'epoch': epoch,
                                                              'loss': loss_per_epoch,
                                                              'valid_precision': precision,
                                                              'valid_recall': recall,
                                                              'valid_fscore': f1_score,
                                                              'time': time.time() - start}), ignore_index=True)

        print('{}epoch\nloss: {}\nvalid: {}\ntime: {} sec.\n'.format(epoch, loss_per_epoch, f1_score, time.time() - start))
    checkpoint(epoch, model, MODEL_PATH, batch_size=opt.batch_size, use_gpu=opt.gpu)
    df_epoch_results.to_csv(os.path.join(RESULT_PATH, 'result_epoch_{}.csv'.format(MODEL_PATH.split('/')[-1])), float_format='%.3f')
