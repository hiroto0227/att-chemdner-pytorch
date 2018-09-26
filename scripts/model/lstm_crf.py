import torch
import torch.nn as nn
from torchcrf import CRF


class LSTMCRFTagger(nn.Module):
    def __init__(self, vocab_dim, tag_dim, embed_dim=300, hidden_dim=1000, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        
        self.embed = nn.Embedding(vocab_dim, embed_dim) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.crf = CRF(self.tag_dim)

    def init_hidden(self):
        return (torch.zeros(2, self.batch_size, self.hidden_dim // 2),
                torch.zeros(2, self.batch_size, self.hidden_dim // 2))

    def forward(self, x, y):
        self.hidden = self.init_hidden()
        embeds = self.embed(x)
        print('\nembed_size: {}'.format(embeds.size()))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        print('lstm_out_size: {}'.format(lstm_out.size()))
        lstm_feats = self.hidden2tag(lstm_out)
        print('lstm_feats: {}'.format(lstm_feats.size()))
        log_likelihood = self.crf(lstm_feats, y)
        return log_likelihood

    def decode(self, x):
        self.hidden = self.init_hidden()
        embeds = self.embed(x)
        print('\nembed_size: {}'.format(embeds.size()))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        print('lstm_out_size: {}'.format(lstm_out.size()))
        lstm_feats = self.hidden2tag(lstm_out)
        print('lstm_feats: {}'.format(lstm_feats.size()))
        decoded = self.crf.decode(lstm_feats)
        print('decoded: {}'.format(decoded.shape))
