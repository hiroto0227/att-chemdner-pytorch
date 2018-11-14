import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torchcrf import CRF
import sys
sys.path.append('..')
from utils import get_variable


class CharLSTM(nn.Module):
    def __init__(self, char_vocab_dim, char_embed_dim, char_lstm_dim, use_gpu=True):
        super().__init__()
        self.char_lstm_dim = char_lstm_dim
        self.use_gpu = use_gpu
        
        self.char_embed = nn.Embedding(char_vocab_dim, char_embed_dim)
        self.char_lstm_f = nn.LSTM(char_embed_dim, char_lstm_dim)
        self.char_lstm_b = nn.LSTM(char_embed_dim, char_lstm_dim)

        if use_gpu:
            self.char_embed.cuda()
            self.char_lstm_f.cuda()
            self.char_lstm_b.cuda()

    def init_char_lstm_hidden(self, char_size):
        return (get_variable(torch.zeros(1, char_size, self.char_lstm_dim), use_gpu=self.use_gpu),
                get_variable(torch.zeros(1, char_size, self.char_lstm_dim), use_gpu=self.use_gpu))

    def forward(self, char_id_seq):
        char_lstm_outs = []
        seq_len, batch_size, max_char_len = char_id_seq.shape
        for char_ids in char_id_seq:
            char_ids_b = get_variable(torch.from_numpy(np.flip(char_ids, 1).copy()), use_gpu=self.use_gpu)
            self.char_lstm_hidden_f = self.init_char_lstm_hidden(max_char_len)
            self.char_lstm_hidden_b = self.init_char_lstm_hidden(max_char_len)
            embed_f = self.char_embed(char_ids)  # (bs, max_char_len, char_embed_dim)
            embed_b = self.char_embed(char_ids_b)  # (bs, max_char_len, char_embed_dim)
            char_lstm_f, self.char_lstm_hidden_f = self.char_lstm_f(embed_f, self.char_lstm_hidden_f)  # (bs, max_char_len, char_lstm_dim)
            char_lstm_b, self.char_lstm_hidden_b = self.char_lstm_b(embed_b, self.char_lstm_hidden_b)  # (bs, max_char_len, char_lstm_dim)
            char_lstm_outs.append(torch.cat((char_lstm_f[:, -1], char_lstm_b[:, -1]), dim=1))  # (bs, char_lstm_dim * 2)
        char_lstm_outs = torch.cat(char_lstm_outs).view(seq_len, batch_size, -1)  # (seq_length, bs, char_lstm_dim * 2)
        return char_lstm_outs


class BiLSTMLSTMCRF(nn.Module):
    def __init__(self, word_vocab_dim, char_vocab_dim, label_dim, word_embed_dim, char_embed_dim, word_lstm_dim, char_lstm_dim, batch_size, pretrain_embed, training, use_gpu=True):
        super().__init__()
        self.batch_size = batch_size
        self.training = training
        self.use_gpu = use_gpu
        self.word_lstm_dim = word_lstm_dim
        self.char_lstm_dim = char_lstm_dim
        char_tanh_dim = char_lstm_dim
        self.char_tanh_dim = char_tanh_dim

        self.char_lstm = CharLSTM(char_vocab_dim, char_embed_dim, char_lstm_dim, use_gpu=use_gpu)
        self.word_embed = nn.Embedding(word_vocab_dim, word_embed_dim, )
        if pretrain_embed is not None:
            pass
            #self.word_embed.weight.data.copy_(torch.from_numpy(pretrain_embed))
        self.word_lstm = nn.LSTM(word_embed_dim + char_lstm_dim * 2, word_lstm_dim, bidirectional=True)
        self.tanh = nn.Linear(word_lstm_dim * 2, label_dim)
        self.crf = CRF(label_dim)

        if use_gpu:
            self.word_embed.cuda()
            self.word_lstm.cuda()
            self.tanh.cuda()
            self.crf.cuda()

    def init_word_hidden(self):
        return (get_variable(torch.zeros(2, self.batch_size, self.word_lstm_dim), use_gpu=self.use_gpu),
                get_variable(torch.zeros(2, self.batch_size, self.word_lstm_dim), use_gpu=self.use_gpu))

    def _forward(self, word, char):
        self.word_lstm_hidden = self.init_word_hidden()
        char_lstm_out = self.char_lstm(char)  # (seq_length, bs, char_hidden_dim)
        word_embeds = self.word_embed(word)  # (seq_length, bs, word_embed_dim)
        char_word_embed = torch.cat((word_embeds, char_lstm_out), dim=2)
        word_lstm_out, self.word_lstm_hidden = self.word_lstm(char_word_embed, self.word_lstm_hidden)  # (seq_length, bs, word_hidden_dim)
        out = self.tanh(word_lstm_out)  # (seq_length, bs, tag_dim)
        return out

    def loss(self, word, char, label):
        out = self._forward(word, char)
        log_likelihood = self.crf(out, label)
        # log_likelihoodを最大にすれば良いが、最小化するので-1をかけている。
        return -1 * log_likelihood

    def forward(self, word, char):
        out = self._forward(word, char)
        decoded = torch.FloatTensor(self.crf.decode(out))
        return decoded
