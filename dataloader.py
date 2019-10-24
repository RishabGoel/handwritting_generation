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
        self.X = X
        self.y = y

        # self.X, _ = self.pad_data(X, self.lens, self.max_len)
        # self.y, self.y_mask = self.pad_data(y, self.lens, self.max_len)
        
        print(self.X.shape, self.y.shape)


    def pad_data(self, data, lens, max_len):
        seq_tensor = np.zeros((data.shape[0], max_len, data[0].shape[-1]))
        mask = np.zeros((data.shape[0], max_len))
        # import pdb; pdb.set_trace()        
        
        for i in range(data.shape[0]):
            seq_tensor[i, :lens[i]] = data[i]
            mask[i, :lens[i]] = 1

        # import pdb; pdb.set_trace()

        return torch.tensor(seq_tensor), torch.tensor(mask)


    def last_batch(self):
        if self.cur_idx > self.X.shape[0]:
            return False
        else:
            return True

    def next_batch(self):
        'Generates one sample of data'
        if self.cur_idx+self.bs>self.X.shape[0]:
            X_batch = self.X[self.cur_idx:]
            y_batch = self.y[self.cur_idx:]

            lens = self.lens[self.cur_idx:]
            X_batch,_ = self.pad_data(X_batch, lens, max(lens))
            y_batch, y_mask = self.pad_data(y_batch, lens, max(lens))
            # tmp_data = torch.tensor(self.X[self.cur_idx:]), torch.tensor(self.y[self.cur_idx:]), torch.tensor(self.y_mask[self.cur_idx:]), torch.tensor(self.lens[self.cur_idx:]).long()
        else:
            # tmp_data = torch.tensor(self.X[self.cur_idx:self.cur_idx + self.bs]), torch.tensor(self.y[self.cur_idx:self.cur_idx + self.bs]), torch.tensor(self.y_mask[self.cur_idx:self.cur_idx+self.bs]), torch.tensor(self.lens[self.cur_idx:self.cur_idx+self.bs]).long()
            X_batch = self.X[self.cur_idx:self.cur_idx + self.bs]
            y_batch = self.y[self.cur_idx:self.cur_idx + self.bs]

            lens = self.lens[self.cur_idx:self.cur_idx + self.bs]
            # print([i for i in range(len(lens)) if lens[i]==max(lens)])
            X_batch,_ = self.pad_data(X_batch, lens, max(lens))
            y_batch, y_mask = self.pad_data(y_batch, lens, max(lens))

        lens = torch.tensor(lens).long()
        self.cur_idx += self.bs

        return X_batch, y_batch, y_mask, lens