import numpy as np

import torch
from torch.utils import data
from torch.autograd import Variable

class Dataset():
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y,  y_mask, text, text_mask):
        
        self.cur_idx = 0
        self.bs = bs
        
        self.X = torch.Tensor(X).float()
        self.y = torch.Tensor(y).float()
        self.y_mask = torch.Tensor(y_mask).float()
        self.text = torch.Tensor(text).float()
        self.text_mask = torch.Tensor(text_mask).float()

        # self.X, _ = self.pad_data(X, self.lens, self.max_len)
        # self.y, self.y_mask = self.pad_data(y, self.lens, self.max_len)
        

    def last_batch(self):
        if self.cur_idx >= self.X.shape[0]:
            return False
        else:
            return True

    def next_batch(self):
        'Generates one sample of data'
        if self.cur_idx+self.bs>self.X.shape[0]:
            X_batch = self.X[self.cur_idx:]
            y_batch = self.y[self.cur_idx:]
            y_mask_batch = self.y_mask[self.cur_idx:]
            
            text_batch = self.text[self.cur_idx:]
            text_mask_batch = self.text_mask[self.cur_idx:]
            # tmp_data = torch.tensor(self.X[self.cur_idx:]), torch.tensor(self.y[self.cur_idx:]), torch.tensor(self.y_mask[self.cur_idx:]), torch.tensor(self.lens[self.cur_idx:]).long()
        else:
            # tmp_data = torch.tensor(self.X[self.cur_idx:self.cur_idx + self.bs]), torch.tensor(self.y[self.cur_idx:self.cur_idx + self.bs]), torch.tensor(self.y_mask[self.cur_idx:self.cur_idx+self.bs]), torch.tensor(self.lens[self.cur_idx:self.cur_idx+self.bs]).long()
            X_batch = self.X[self.cur_idx:self.cur_idx + self.bs]
            y_batch = self.y[self.cur_idx:self.cur_idx + self.bs]
            y_mask_batch = self.y_mask[self.cur_idx:self.cur_idx + self.bs]
            
            text_batch = self.text[self.cur_idx:self.cur_idx + self.bs]
            text_mask_batch = self.text_mask[self.cur_idx:self.cur_idx + self.bs]

        self.cur_idx += self.bs

        return X_batch, y_batch, y_mask_batch, text_batch, text_mask_batch