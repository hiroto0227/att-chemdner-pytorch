import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torchcrf import CRF


class LSTMTagger(nn.Module):
    def __init__(self, vocab_dim, tag_dim, embed_dim=300, hidden_dim=1000, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.hidden = self.init_hidden()
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.tag_dim = tag_dim
        self.crf = CRF(tag_dim)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x):
        self.hidden = self.init_hidden()
        embeds = self.embed(x) # (batch_size, seq_length, embed_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) #(batch_size, seq_length, hidden_dim)
        tag_space = self.hidden2tag(lstm_out) # (batch_size, seq_length, tag_size)
        emissions = F.log_softmax(tag_space, dim=2) # (batch_size, seq_length, tag_size)
        return emissions
