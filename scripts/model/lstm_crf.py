import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.autograd import Variable as Var


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tag_size, EMBED_DIM=300, HIDDEN_DIM=1000, BATCH_SIZE=32):
        super().__init__()
        self.batch_size = BATCH_SIZE
        self.embed_dim = EMBED_DIM
        self.hidden_dim = HIDDEN_DIM
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM)
        self.hidden = self.init_hidden()
        self.hidden2tag = nn.Linear(HIDDEN_DIM, tag_size)
        self.tag_size = tag_size
        self.crf = CRF(tag_size)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x, y):
        self.hidden = self.init_hidden()
        embeds = self.embed(x)
        #print('embed_size: {}'.format(embeds.size()))
        lstm_out, self.hidden = self.lstm(
            embeds.view(-1, self.batch_size, self.embed_dim), self.hidden)
        #print('lstm_out_size: {}'.format(lstm_out.size()))
        tag_space = self.hidden2tag(lstm_out)
        #print('tag_space: {}'.format(tag_space.size()))
        emissions = F.log_softmax(tag_space, dim=1)
        #print('emissions: {}'.format(emissions.size()))
        tag = self.crf(emissions, y)
        return tag