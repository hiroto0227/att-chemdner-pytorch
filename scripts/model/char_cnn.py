import torch
import torch.nn as nn
import torch.nn.functional as F


class  CharCNN(nn.Module):
    
    def __init__(self, alphabet_size, embedding_dim=100, hidden_dim=50, dropout=0.5, gpu=True):
        super(CharCNN, self).__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_cnn1 = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool1d(4)
        self.char_cnn2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool1d(4)
        self.char_cnn3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool1d(2)
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_cnn1 = self.char_cnn1.cuda()
            self.pool1 = self.pool1.cuda()
            self.char_cnn2 = self.char_cnn2.cuda()
            self.pool2 = self.pool2.cuda()
            self.char_cnn3 = self.char_cnn3.cuda()
            self.pool3 = self.pool3.cuda()

    def forward(self, x):
        char_embeds = self.char_drop(self.char_embeddings(x))
        #print('char_embeds: {}'.format(char_embeds.view(-1, char_embeds.size(2), char_embeds.size(3)).shape))
        char_cnn_out = self.char_cnn1(char_embeds.view(-1, char_embeds.size(2), char_embeds.size(3)).transpose(2, 1))
        #print('char_cnn_out: {}'.format(char_cnn_out.shape))
        char_cnn_out = self.pool1(char_cnn_out)
        #print('char_cnn_out: {}'.format(char_cnn_out.shape))
        char_cnn_out = self.char_cnn2(char_cnn_out)
        #print('char_cnn_out: {}'.format(char_cnn_out.shape))
        char_cnn_out = self.pool2(char_cnn_out)
        #print('char_cnn_out: {}'.format(char_cnn_out.shape))
        char_cnn_out = self.char_cnn3(char_cnn_out)
        #print('char_cnn_out: {}'.format(char_cnn_out.shape))
        char_cnn_out = self.pool3(char_cnn_out)
        #print('char_cnn_out: {}'.format(char_cnn_out.shape))
        return char_cnn_out.view(x.size(0), x.size(1), -1)
