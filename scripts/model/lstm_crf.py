import torch
import torch.nn as nn
from torchcrf import CRF
import sys
sys.path.append('..')
from utils import get_variable


class LSTMCRFTagger(nn.Module):
    def __init__(self, vocab_dim, tag_dim, embed_dim=300, hidden_dim=1000, batch_size=32, use_gpu=True):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.use_gpu = use_gpu
        
        self.embed = nn.Embedding(vocab_dim, embed_dim) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.crf = CRF(self.tag_dim)

    def init_hidden(self):
        return (get_variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2), use_gpu=self.use_gpu),
                get_variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2), use_gpu=self.use_gpu))

    def loss(self, x, y):
        mask = get_variable(torch.autograd.Variable(x.data.gt(0)), use_gpu=self.use_gpu)
        self.hidden = self.init_hidden()
        embeds = self.embed(x)
        #print('\nembed_size: {}'.format(embeds.size()))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        #print('lstm_out_size: {}'.format(lstm_out.size()))
        lstm_feats = self.hidden2tag(lstm_out)
        #print('lstm_feats: {}'.format(lstm_feats.size()))
        log_likelihood = self.crf(lstm_feats, y, mask=mask)
        return - log_likelihood

    def forward(self, x):
        self.hidden = self.init_hidden()
        embeds = self.embed(x)
        #print('\nembed_size: {}'.format(embeds.size()))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        #print('lstm_out_size: {}'.format(lstm_out.size()))
        lstm_feats = self.hidden2tag(lstm_out)
        #print('lstm_feats: {}'.format(lstm_feats.size()))
        decoded = torch.FloatTensor(self.crf.decode(lstm_feats))
        #print('decoded: {}'.format(decoded.shape))
        return decoded
