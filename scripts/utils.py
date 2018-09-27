import os
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
