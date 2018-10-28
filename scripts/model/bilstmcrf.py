import torch
import torch.nn as nn
from torchcrf import CRF
import sys
sys.path.append('..')
from utils import get_variable


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_dim, tag_dim, embed_dim=300, hidden_dim=1000, batch_size=32, num_layers=1, use_gpu=True):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        self.embedding = nn.Embedding(vocab_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True)
        self.hidden = self.init_hidden()
        self.hidden2emb = nn.Linear(hidden_dim, embed_dim)
        self.hidden2tag = nn.Linear(embed_dim, tag_dim)
        self.crf = CRF(self.tag_dim)

    def init_hidden(self):
        if self.training:
            return (get_variable(torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_dim // 2), use_gpu=self.use_gpu),
                    get_variable(torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_dim // 2), use_gpu=self.use_gpu))
        else:
            return (get_variable(torch.zeros(2 * self.num_layers, 1, self.hidden_dim // 2), use_gpu=self.use_gpu),
                    get_variable(torch.zeros(2 * self.num_layers, 1, self.hidden_dim // 2), use_gpu=self.use_gpu))
       
    def _forward(self, x):
        self.hidden = self.init_hidden()
        embed = self.embedding(x)
        lstm_out, self.hidden = self.lstm(embed, self.hidden)
        emb = self.hidden2emb(lstm_out)
        out = self.hidden2tag(emb)
        return out

    def loss(self, x, y):
        out = self._forward(x)
        log_likelihood = self.crf(out, y)
        # log_likelihoodを最大にすれば良いが、最小化するので-1をかけている。
        return -1 * log_likelihood

    def forward(self, x):
        out = self._forward(x)
        decoded = torch.FloatTensor(self.crf.decode(out))
        return decoded
