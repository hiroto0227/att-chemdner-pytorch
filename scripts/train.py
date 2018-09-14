import os, sys
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
import torch
import torchtext
from torchtext.data import BucketIterator
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from dataset import ChemdnerDataset
from model.lstm import LongShortTermMemory

if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 4
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '../models/lstm_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    train_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/train.csv'))
    token2id, label2id = train_dataset.make_vocab()
    #valid_dataset = ChemdnerDataset(path=os.path.join(CURRENT_DIR, '../datas/processed/valid.csv'))
    
    model = LongShortTermMemory(vocab_size=len(token2id), tag_size=len(label2id), BATCH_SIZE=BATCH_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_sum = 0

    for epoch in range(EPOCHS):
        start = time.time()
        loss_per_epoch = 0
        for batch in tqdm(BucketIterator(train_dataset, batch_size=BATCH_SIZE, shuffle=True, repeat=False)):
            try:
                output = model(batch.text)
                one_hot_labels = torch.LongTensor(batch.label.size(0), batch.label.size(1), len(label2id)).zero_()
                one_hot_labels.scatter_(2, batch.label.unsqueeze(2), 1)
                # (seq_length, batch_size, tag_size) -> (seq_length * batch_size, tag_size)
                loss = F.nll_loss(output.view(-1, len(label2id)), batch.label.view(-1))
                loss.backward()
                optimizer.step()
                print('loss: {}'.format(loss))
                loss_per_epoch += float(loss)
            except KeyboardInterrupt:
                print('model saved!!')
                torch.save(model.state_dict(), MODEL_PATH)
                sys.exit(1)
        print('{}epoch\nloss: {}\ntime: {} sec.'.format(epoch, loss_per_epoch, time.time() - start))
    print('model saved!!')
    torch.save(model.state_dict(), MODEL_PATH)

        