import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    def __init__(self, vocab_dim, embed_dim=100, hidden_dim=100, batch_size=10, use_gpu=True):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_dim, embed_dim)
        #self.embedding.data.uniform_(-0.5 / vocab_dim, 0.5 / vocab_dim)
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        #self.lstm.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        self.decode = nn.Linear(hidden_dim, embed_dim)
        #self.decode.data.uniform(-0.5 / hidden_dim, 0.5 / hidden_dim)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, self.hidden = self.lstm(embed)
        decoded = self.decode(lstm_out)
        return decoded
