import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable as Var
import time


class Att_LSTM(nn.Module):
    def __init__(self, vocab_dim, tag_dim, embed_dim=100, hidden_dim=200, att_hidden_dim=100, batch_size=32):
        super(Att_LSTM, self).__init__() 
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim
        self.tag_dim = tag_dim

        self.embed = nn.Embedding(vocab_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.attention = Attention(batch_size, hidden_dim, att_hidden_dim)
        self.hidden = self.init_hidden()
        self.att2tag = nn.Linear(att_hidden_dim, tag_dim)
        self.tag_dim = tag_dim

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x):
        self.hidden = self.init_hidden()
        embedding = self.embed(x)
        lstm_out, self.hidden = self.lstm(embedding, self.hidden) # eq.1
        att_out = self.attention(embedding, lstm_out)
        tag_space = self.att2tag(att_out) # eq.9
        # To Do: CRF層を追加 eq.10
        emissions = F.log_softmax(tag_space, dim=1)
        return emissions


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, att_hidden_dim):
        super(Attention, self).__init__()
        self.weight_sim = nn.Linear(embed_dim, 1)
        self.weight_global = nn.Linear(hidden_dim * 2, att_hidden_dim)
    
    def forward(self, x, h):
        """
        input x: size=(seq_length, batch_size, embed_size)
        return out: size=(seq_length, batch_size, out_size)
        """
        print('\n-- Attention Forward --')
        start = time.time()
        seq_len = x.shape[0]
        A = self._calc_A(x) # eq.5, eq.6
        G = torch.zeros(h.shape[0], h.shape[1], h.shape[2])
        for t in range(seq_len):
            G[t] = sum([A[t, :, j] * h[j, :, :] for j in range(h.size(0))]) # eq.7
        Z = torch.tanh(self.weight_global(torch.cat((G, h), dim=2))) # eq.8
        print('{} sec\n'.format(time.time() - start))
        return Z
        
    def _calc_A(self, x):
        """equation 6 in paperc
        input x: size=(seq_length, batch_size, embed_size)
        return A: size=(seq_length, batch_size, seq_length, 1)
        """
        seq_len = x.size(0)
        A = torch.zeros(seq_len, x.size(1), seq_len, 1)
        for t in range(seq_len):
            for j in range(seq_len):
                score = F.pairwise_distance(x[t], x[j])
                A[t, :, j, :] = F.softmax(self.weight_sim(score), 0).unsqueeze(1)
        return A