import os
<<<<<<< HEAD
import numpy as np
=======
from collections import deque
>>>>>>> 325867741129368dffe36d86e449df0979a23fa7
import torch
from torch.autograd import Variable
from labels import UNK, PAD, COMMA


def get_variable(tensor, use_gpu=False, **kwargs):
    if torch.cuda.is_available() and use_gpu:
        result = Variable(tensor.cuda(), **kwargs)
    else:
        result = Variable(tensor, **kwargs)
    return result


def checkpoint(epoch, model, model_path, batch_size, interrupted=False, use_gpu=False):
    print('model saved!!')
    gpu_flag = 'gpu' if use_gpu and torch.cuda.is_available() else 'cpu'
    interrupted_flag = 'interrupted' if interrupted else ''
    model_dir, model_name = os.path.split(model_path)
    model_name = '{}_{}_{}ep_{}bs_{}.pth'.format(gpu_flag, model_name, epoch, batch_size, interrupted_flag)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))


<<<<<<< HEAD
def make_subwords_from_token_batches(token_batch, id2token, id2char, tokenize, batch_first=True):
    """入力のtokenをsubword化して返す。
    input:token_batch
        size=(batch_size, seq_len, 1)
    output:subwords
        size=(batch_size, seq_len, max_subword_len)
    """
    char2id = {c:i for i, c in enumerate(id2char)}
    token_ids_batch = token_batch #(batch_size, seq_len, 1)
    subword_batch = []
    max_subword_length = 0
    for token_ids in token_ids_batch:
        tokens = [id2token[token_id] for token_id in token_ids] #(seq_len)
        subwords = []
        for i, token in enumerate(tokens):
            if not token in [UNK, PAD, '<pad>', COMMA]:
                subwords.append(np.array([char2id[subword] for subword in tokenize(token)]))
            else:
                subwords.append(np.array([char2id[token]]))
            max_subword_length = len(subwords[-1]) if len(subwords[-1]) > max_subword_length else max_subword_length
        subword_batch.append(subwords) # (batch_size, seq_len, subword_size)
    padded_subwords = [[np.pad(subwords, mode='constant', pad_width=(0, max_subword_length - len(subwords)))
                        for subwords in batch] for batch in subword_batch]
    return torch.LongTensor(padded_subwords) # (batch_size, seq_len, subword_size)  
=======
class EarlyStop():
    def __init__(self, stop_not_rise_num=5, threshold_rate=0.1):
        self.stop_not_rise_num = stop_not_rise_num
        self.threshold_rate = threshold_rate
        self.valid_scores_que = deque([0 for i in range(self.stop_not_rise_num)])

    def is_end(self, score):
        first_score = self.valid_scores_que.popleft()
        self.valid_scores_que.append(score)
        if first_score - score < self.threshold_rate:
            return False
        else:
            return True
>>>>>>> 325867741129368dffe36d86e449df0979a23fa7
