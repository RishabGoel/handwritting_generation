import numpy as np

import torch
from torch.utils import data
from torch.autograd import Variable

class Dataset():
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y, bs):
        'Initialization'
        self.cur_idx = 0
        self.bs = bs
        self.lens = [sample.shape[0] for sample in X]
        self.max_len = max(self.lens)
        
        self.X, _ = self.pad_data(X, self.lens, self.max_len)
        self.y, self.y_mask = self.pad_data(y, self.lens, self.max_len)
        
        print(self.X.shape, self.y.shape)


    def pad_data(self, data, lens, max_len):
        seq_tensor = np.zeros((data.shape[0], max_len, data[0].shape[-1]))
        mask = np.zeros((data.shape[0], max_len, data[0].shape[-1]))
        # import pdb; pdb.set_trace()        
        
        for i in range(data.shape[0]):
            seq_tensor[i, :lens[i]] = data[i]
            mask[i, :lens[i]] = 1

        # import pdb; pdb.set_trace()

        return seq_tensor, mask


    def last_batch(self):

    def next_batch(self):
        'Generates one sample of data'
        if self.cur_idx+self.bs>self.X.shape[0]:
            tmp_data = torch.tensor(self.X[self.cur_idx:]), torch.tensor(self.y[self.cur_idx:]), torch.tensor(self.y_mask[self.cur_idx:]), self.lens[self.cur_idx:]
        else:
            tmp_data = torch.tensor(self.X[self.cur_idx:self.cur_idx + self.bs]), torch.tensor(self.y[self.cur_idx:self.cur_idx + self.bs]), torch.tensor(self.y_mask[self.cur_idx:self.cur_idx+self.bs]), self.lens[self.cur_idx:self.cur_idx+self.bs]

        self.cur_idx += self.bs

        return tmp