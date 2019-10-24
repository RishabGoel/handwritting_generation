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
        self.num_mixtures = args.num_mixtures
        self.model = Model(self.inp_dim, self.hidden_units, self.out_dim, self.num_layers)
    

    def get_likelihood(self, e, ro, pi, mu, sigma, y_true):
        mu_shp = mu.shape

        mu = mu.unsqueeze(3).reshape(mu_shp[0], mu_shp[1], mu_shp[2]//self.num_mixtures , self.num_mixtures)
        
        mu_x = mu[:,:,0,:]
        mu_y = mu[:,:,1,:] 
        
        sigma = sigma.unsqueeze(3).reshape(mu_shp[0], mu_shp[1], mu_shp[2]//self.num_mixtures , self.num_mixtures)
        sigma_x = sigma[:,:,0,:]
        sigma_y = sigma[:,:,1,:]

        reci_sigma_xy = torch.reciprocal(sigma_x * sigma_y)
        reci_sigma_xx = torch.reciprocal(sigma_x * sigma_x)
        reci_sigma_yy = torch.reciprocal(sigma_y * sigma_y)
        

        cood_x = y_true[:,:,1:2]
        cood_y = y_true[:,:,2:]
        
        exp_denom_ro = torch.reciprocal(2 * (1 - ro * ro))
        norm_denom_ro = torch.reciprocal(2 * np.pi * torch.sqrt(1-ro)) * reci_sigma_xy

        z_xx = (cood_x - mu_ x) * (cood_x - mu_ x) * reci_sigma_xx
        z_yy = (cood_y - mu_ y) * (cood_y - mu_ y) * reci_sigma_yy
        z_xy = -2 * ro * (cood_y - mu_ y) * (cood_x - mu_ x) * reci_sigma_xy

        z = z_xx + z_yy + z_xy

        N = torch.sum(norm_denom_ro * torch.exp(-1 * z * exp_denom_ro) * pi, dim = 2)
        N = -1 * torch.log(N)
        import pdb; pdb.set_trace()

        return N

    def fit(self, X, y):

        optimizer = torch.optim.RMSprop(self.model.parameters())
        bce = nn.BCELoss(reduction=None)

        for epoch in range(self.epochs):
            training_set = Dataset(X, y, self.bs)
            while training_set.last_batch():
                # Transfer to GPU
                X_batch, y_batch, y_mask_batch, lens_batch = training_set.next_batch()
                X_batch, y_batch, y_mask_batch, lens_batch = X_batch.to(self.device), y_batch.to(self.device), y_mask_batch.to(self.device), lens_batch.to(self.device)
                # print(training_set.cur_idx)
                e,ro,pi,mu,sigma = self.model(X_batch, lens_batch)
                import pdb; pdb.set_trace()
                
                N = self.get_likelihood(e,ro,pi,mu,sigma, y_batch)
                N = N.reshape(-1)
                
                y_ber_truth = y_batch[:,:,0].reshape(-1)
                
                e = e.reshape(-1)

                y_mask_batch = y_mask_batch.reshape(-1)

                e_loss = bce(e, y_ber_truth.long())
                loss_sum = torch.sum(N*y_mask_batch) + torch.sum(e_loss * y_mask_batch)
                loss = loss_sum / torch.sum(lens_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    print(epoch, loss.data[0])


