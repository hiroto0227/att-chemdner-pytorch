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
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True)
        #self.lstm.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        self.hidden2emb = nn.Linear(hidden_dim, embed_dim)
        #self.decode.data.uniform(-0.5 / hidden_dim, 0.5 / hidden_dim)

    def init_hidden(self):
        return (torch.zeros(2, self.batch_size, self.hidden_dim),
                torch.zeros(2, self.batch_size, self.hidden_dim))

    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, self.hidden = self.lstm(embed)
        decoded = self.hidden2emb(lstm_out)
        return decoded
