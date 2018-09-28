import os
from collections import deque
import torch
from torch.autograd import Variable


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
