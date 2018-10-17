import torch
import torch.nn as nn
from torchcrf import CRF
import sys
sys.path.append('..')
from utils import get_variable


class BiLSTMBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_dim, batch_size, num_layers=1, use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True)
        self.hidden = self.init_hidden()
        self.h2out = nn.Linear(hidden_dim, out_dim)

    def init_hidden(self):
        return (get_variable(torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_dim // 2), use_gpu=self.use_gpu),
                get_variable(torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_dim // 2), use_gpu=self.use_gpu))

    def forward(self, x):
        # mask = get_variable(torch.autograd.Variable(x.data.gt(0)), use_gpu=self.use_gpu)
        self.hidden = self.init_hidden()
        h, self.hidden = self.lstm(x, self.hidden)
        out = self.h2out(h)
        return out


class TagSpaceConcatMultiSubwordSequences(nn.Module):
    def __init__(self, char_vocab_dim, char_embed_dim, char_hidden_dim, sub_vocab_dims, sub_embed_dims, sub_hidden_dims, tag_dim, batch_size, use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu
        self.embed_char = nn.Embedding(char_vocab_dim, char_embed_dim)
        self.embed_subs = [nn.Embedding(vocab_dim, embed_dim, tag_dim).cuda() for vocab_dim, embed_dim in zip(sub_vocab_dims, sub_embed_dims)]
        self.lstm_block_char = BiLSTMBlock(char_embed_dim, char_hidden_dim, tag_dim, batch_size)
        self.lstm_block_subs = [BiLSTMBlock(embed_dim, hidden_dim, tag_dim, batch_size).cuda() for embed_dim, hidden_dim in zip(sub_embed_dims, sub_hidden_dims)]
        self.outs2tag = nn.Linear(tag_dim * (1 + len(sub_hidden_dims)), tag_dim)
        self.crf = CRF(tag_dim)

    def _forward(self, x_char, x_subs):
        embed_char = self.embed_char(x_char)
        out_char = self.lstm_block_char(embed_char)
        out_subs = []
        for x_sub, embed_sub, lstm_block_sub in zip(x_subs, self.embed_subs, self.lstm_block_subs):
            embed = embed_sub(x_sub)
            out_subs.append(lstm_block_sub(embed))
        outs = torch.cat([out_char] + out_subs, dim=2)
        tag = self.outs2tag(outs)
        return tag

    def loss(self, x_char, x_subs, y):
        z = self._forward(x_char, x_subs)
        log_likelihood = self.crf(z, y)
        # log_likelihoodを最大にすれば良いが、最小化するので-1をかけている。
        return -1 * log_likelihood

    def forward(self, x_char, x_subs):
        z = self._forward(x_char, x_subs)
        decoded = torch.FloatTensor(self.crf.decode(z))
        return decoded
