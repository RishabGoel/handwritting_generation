import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from models.model import *
from dataloader import Dataset

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args):
        
        self.lr = args.lr
        self.inp_dim = args.inp_dim
        self.hidden_units = args.hidden_units
        self.out_dim = args.out_dim
        self.bs = args.bs
        self.epochs = args.epochs
        self.num_layers = args.num_layers
        self.device = 'cpu'
        self.model = Model(self.inp_dim, self.hidden_units, self.out_dim, self.num_layers)

    def loss(self):
        pass

    def fit(self, X, y):

        for epoch in range(self.epochs):
            training_set = Dataset(X, y, self.bs)
            while training_set.last_batch():
                # Transfer to GPU
                # import pdb; pdb.set_trace()
                X, y = X.to(device), y.to(device)