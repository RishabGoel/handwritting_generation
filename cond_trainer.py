import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from models.cond_model import *
from dataloader import Dataset
import utils.helper as utils
# import numpy
from matplotlib import pyplot



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
        self.model = Model(self.inp_dim, self.hidden_units, self.out_dim, self.num_layers, self.num_mixtures).to(self.device)
        self.validation_score = np.inf
        self.valid_low_count = 0
        self.valid_low_count_max = 5
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.bce = nn.BCELoss(reduction='none')
        self.path = 'model'
    

    def get_likelihood(self, e, ro, pi, mu, sigma, y_true):
        mu_shp = mu.shape
        # import pdb; pdb.set_trace()
        mu = mu.unsqueeze(3).reshape(mu_shp[0], mu_shp[1], mu_shp[2]//self.num_mixtures , self.num_mixtures)
        
        mu_x = mu[:,:,0,:]
        mu_y = mu[:,:,1,:] 
        
        # import pdb; pdb.set_trace()
        sigma = sigma.unsqueeze(3).reshape(mu_shp[0], mu_shp[1], mu_shp[2]//self.num_mixtures , self.num_mixtures)
        sigma_x = sigma[:,:,0,:]
        sigma_y = sigma[:,:,1,:]


        # import pdb; pdb.set_trace()
        reci_sigma_xy = torch.reciprocal(sigma_x * sigma_y)
        reci_sigma_xx = torch.reciprocal(sigma_x * sigma_x)
        reci_sigma_yy = torch.reciprocal(sigma_y * sigma_y)
        
        # import pdb; pdb.set_trace()
        cood_x = y_true[:,:,1:2].float()
        cood_y = y_true[:,:,2:].float()
        # import pdb; pdb.set_trace()
        exp_denom_ro = torch.reciprocal(2 * (1 - ro * ro))
        # import pdb; pdb.set_trace()
        norm_denom_ro = torch.reciprocal(2 * np.pi * torch.sqrt(1-ro*ro)) * reci_sigma_xy
        # import pdb; pdb.set_trace()
        z_xx = (cood_x - mu_x) * (cood_x - mu_x) * reci_sigma_xx
        z_yy = (cood_y - mu_y) * (cood_y - mu_y) * reci_sigma_yy
        z_xy = -2 * ro * (cood_y - mu_y) * (cood_x - mu_x) * reci_sigma_xy
        # import pdb; pdb.set_trace()
        z = z_xx + z_yy + z_xy
        # import pdb; pdb.set_trace()
        exp_term = torch.exp(-1 * z * exp_denom_ro)
        N = torch.sum(norm_denom_ro * exp_term * pi, dim = 2)
        # import pdb; pdb.set_trace()
        N = -1 * torch.log(N + 1e-20)
        # import pdb; pdb.set_trace()

        return N

    def fit(self, X, y, test_X, test_y, train_mean, train_std):
        # plot_stroke(utils.un_normalize_data(X[:1], train_mean, train_std)[0])
        # ll
        # optimizer = torch.optim.RMSprop(self.model.parameters())
        # optimizer = torch.optim.Adam(self.model.parameters())
        # bce = nn.BCELoss(reduction='none')

        for epoch in range(self.epochs):
            training_set = Dataset(X, y, self.bs)
            while training_set.last_batch():
                # Transfer to GPU
                X_batch, y_batch, y_mask_batch, lens_batch = training_set.next_batch()
                X_batch, y_batch, y_mask_batch, lens_batch = X_batch.to(self.device), y_batch.to(self.device), y_mask_batch.to(self.device), lens_batch.to(self.device)
                # print(training_set.cur_idx)
                e,ro,pi,mu,sigma, _, _ = self.model(X_batch, lens_batch)
                import pdb; pdb.set_trace()
                
                N = self.get_likelihood(e,ro,pi,mu,sigma, y_batch)
                N = N.reshape(-1)
                
                y_ber_truth = y_batch[:,:,0].reshape(-1).float()
                
                e = e.reshape(-1)

                y_mask_batch = y_mask_batch.reshape(-1)

                ber_loss = self.bce(e, y_ber_truth)
                loss_sum = torch.sum(N*y_mask_batch) + torch.sum(ber_loss * y_mask_batch)
                loss = loss_sum / torch.sum(lens_batch)
                print(loss, loss_sum)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10)

                self.optimizer.step()
                
            if epoch % 5 == 0:
                to_stop = self.validate(test_X, test_y, epoch)
                self.create_stroke()
                if to_stop:
                    break
            break

    def validate(self, test_x, test_y, epoch_id):
        '''
        This function print the validation loss after specified number of epochs and implements early stopping
        '''
        self.model.eval()
        test_set = Dataset(test_x, test_y, self.bs)
        total_loss = 0.0
        while test_set.last_batch():
            # Transfer to GPU
            X_batch, y_batch, y_mask_batch, lens_batch = test_set.next_batch()
            X_batch, y_batch, y_mask_batch, lens_batch = X_batch.to(self.device), y_batch.to(self.device), y_mask_batch.to(self.device), lens_batch.to(self.device)
            # print(training_set.cur_idx)
            e,ro,pi,mu,sigma, _ = self.model(X_batch, lens_batch)
            # import pdb; pdb.set_trace()
            
            N = self.get_likelihood(e,ro,pi,mu,sigma, y_batch)
            N = N.reshape(-1)
            
            y_ber_truth = y_batch[:,:,0].reshape(-1).float()
            
            e = e.reshape(-1)

            y_mask_batch = y_mask_batch.reshape(-1)

            ber_loss = self.bce(e, y_ber_truth)
            loss_sum = torch.sum(N*y_mask_batch) + torch.sum(ber_loss * y_mask_batch)
            loss = loss_sum / torch.sum(lens_batch)
            total_loss += list(loss.cpu().data.numpy().flatten())[0]
        self.model.train()
        print("Validation loss after ", epoch_id, " epochs : ", loss.data)
        if total_loss < self.validation_score:
            self.validation_score = total_loss
            self.valid_low_count = 0
            torch.save(self.model.state_dict(), self.path+str(epoch_id)+".pt")
        else:
            self.valid_low_count += 1
            print("Early stopping count increased from ", self.valid_low_count-1, ' to ', self.valid_low_count)

        if self.valid_low_count >=self.valid_low_count_max:
            return True
        else:
            return False


