import torch
import torch.nn as nn
from torchcrf import CRF
import sys
sys.path.append('..')
# from utils import get_variable


class MultiSubwordAttentionTagger(nn.Module):
    def __init__(self, char_vocab_dim, sub_vocab_dims, tag_dim, char_embed_dim, sub_embed_dims, batch_size, hidden_dim, use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu

        self.embed_char = nn.Embedding(char_vocab_dim, char_embed_dim)
        self.embed_subs = [nn.Embedding(vocab_dim, embed_dim) for vocab_dim, embed_dim in zip(sub_vocab_dims, sub_embed_dims)]
        self.x2h = nn.Linear(char_embed_dim + sum(sub_embed_dims), hidden_dim)
        self.h2hs = []
        self.attention_subs = [Attention() for i in range(len(sub_vocab_dims))]
        self.hg2z = nn.Linear(hidden_dim * (1 + len(sub_vocab_dims)), tag_dim)
        self.z2zs = []
        self.crf = CRF(tag_dim)

    def _forward(self, x_char, x_subs):
        embed_char = self.embed_char(x_char)
        embed_subs = []
        for i, embed_sub in enumerate(self.embed_subs):
            embed_subs.append(embed_sub(x_subs[i]))
        embed = torch.cat([embed_char] + embed_subs, dim=2)
        h = self.x2h(embed)
        for i, h2h in enumerate(self.h2hs):
            h = h2h(h)
        g_subs = []
        for i, attention_sub in enumerate(self.attention_subs):
            g_subs.append(attention_sub(embed_subs[i], h))
        hg = torch.cat([h] + g_subs, dim=2)
        z = self.hg2z(hg)
        for i, z2z in enumerate(self.z2zs):
            z = self.z2z(z)
        return z

    def loss(self, x_char, x_subs, y):
        z = self._forward(x_char, x_subs)
        log_likelihood = self.crf(z, y)
        # log_likelihoodを最大にすれば良いが、最小化するので-1をかけている。
        return -1 * log_likelihood

    def forward(self, x_char, x_subs):
        z = self._forward(x_char, x_subs)
        decoded = torch.FloatTensor(self.crf.decode(z))
        return decoded


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_scores(self, x):
        """xの総当たりの類似度を求める。"""
        # (B, N ,H) x (B, H, N) = (B, N, N)
        x = x.transpose(1, 0)
        norm = torch.norm(x.float(), dim=2).unsqueeze(2)
        scores = torch.bmm(x / norm, (x / norm).transpose(2, 1))
        return scores  # (B, N_t, N_j)

    def forward(self, x, h):
        """
        input:
            x: (N, batch_size, embed_dim)
            h: (N, batch_size, hidden_dim)
        return:
            y: (N, batch_size, hidden_dim)
        """
        N, B, H = x.size(0), x.size(1), h.size(-1)
        scores = nn.functional.softmax(self._get_scores(x), dim=2)  # (B, Nt, Nj)
        scores = scores.unsqueeze(-1).expand(B, N, N, H)  # (B, Nt, Nj, H)
        scores = scores.transpose(3, 2).view(-1, H, N)
        h = h.transpose(1, 0).contiguous().unsqueeze(1).expand(B, N, N, H).view(-1, N, H)
        g = torch.bmm(scores, h)  # (B, Nt, H, Nj) x (B, Nt, H) -> (B, Nt, H, H)
        g = g.sum(1).view(B, N, -1).transpose(1, 0)
        return g / g.float().norm()  # (Nt, B, H)

        def test_attention(self):
            x = torch.FloatTensor([
                [[1, 1, 1, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 1, 1, 1, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
                [[1, 1, 1, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
                [[0, 0, 0, 1, 1, 1, 0, 0, 0]],
                [[1, 1, 1, 0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 0, 0, 0, 0, 0, 0]],
            ])

            h = torch.FloatTensor([
                [[1, 0, 0]],
                [[0, 1, 0]],
                [[0, 0, 1]],
                [[0, 0, 1]],
                [[1, 0, 0]],
                [[0, 0, 1]],
                [[0, 1, 0]],
                [[1, 0, 0]],
                [[1, 0, 0]],
            ])
            y = self.forward(x, h)
            true_att = torch.FloatTensor([])  # 時間がある時計算してみる。
            assert y == true_att, "Not valid Attention."
