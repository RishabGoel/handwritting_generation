import torch
from torch.utils import data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y
        self.lens = [sample.shape[0] for sample in X]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.lens)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        return self.X[index], self.y[index], self.lens[index]