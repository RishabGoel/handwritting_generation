import torch
import numpy as np
from models.model import  *
from utils.helper import *

class Tester():
    """docstring for Tester"""
    def __init__(self, args, path_to_model, model_type, cond_text = None, model = None):
        
        self.model_type = model_type
        self.cond_text = cond_text
        self.gen_seq_len = args.new_seq_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_layers = args.num_layers
        self.hidden_units = args.hidden_units
        self.num_mixtures = args.num_mixtures
        self.prob_bias = args.prob_bias
        # if model is None:
        #   if model_type == "uncond":
        #       self.model = Model(args.inp_dim, args.hidden_units, args.out_dim, args.num_layers, args.num_mixtures).to(self.device)   
            
        #   self.model.load_state_dict(torch.load(path_to_model))
        # else:
        #   self.model = model

        # self.model.eval()
    
    def generate(self, model, train_mean, train_std):
        # if model is not None:
        self.model = model
        new_seq = np.zeros((self.gen_seq_len, 3))
        new_seq[0] = [0, train_mean[0], train_mean[1]]
        lstm_hc_lst = model.init_hidden_cell_state(1)
        
        # for i in range(self.num_layers):
        #     lstm_hc_lst.append((torch.zeros(1,1,self.hidden_units), torch.zeros(1,1,self.hidden_units)))
        # import pdb;pdb.set_trace()
        for seq_time in range(1, self.gen_seq_len):
            inp = torch.Tensor(new_seq[seq_time-1:seq_time]).unsqueeze(0).to(self.device)
            # import pdb;pdb.set_trace()
            lens = torch.tensor([1]).long().to(self.device)
            # import pdb;pdb.set_trace()
            e,ro,pi,mu,sigma, lstm_hc_lst = self.model(inp, lens, lstm_hc_lst)
            e,ro,pi,mu,sigma = e.cpu(),ro.cpu(), pi.cpu(), mu.cpu(), sigma.cpu()
            # import pdb;pdb.set_trace()
            pi = pi.view(-1)
            # import pdb;pdb.set_trace()
            mix_id = torch.multinomial(pi, 1)
            # import pdb;pdb.set_trace()
            mu_shp = mu.shape
            mu = mu.unsqueeze(3).reshape(mu_shp[0], mu_shp[1], mu_shp[2]//self.num_mixtures , self.num_mixtures)    
            # import pdb;pdb.set_trace()
            mu_x = mu[:,:,0,:].view(-1)
            mu_x = torch.gather(mu_x, 0, mix_id)
            # import pdb;pdb.set_trace()
            mu_y = mu[:,:,1,:].view(-1) 
            mu_y = torch.gather(mu_y, 0, mix_id)
            # import pdb;pdb.set_trace()
            # import pdb; pdb.set_trace()
            sigma = sigma.unsqueeze(3).reshape(mu_shp[0], mu_shp[1], mu_shp[2]//self.num_mixtures , self.num_mixtures)
            # import pdb;pdb.set_trace()
            sigma_x = sigma[:,:,0,:].view(-1)
            sigma_x = torch.gather(sigma_x, 0, mix_id)
            # import pdb;pdb.set_trace()
            sigma_y = sigma[:,:,1,:].view(-1)
            sigma_y = torch.gather(sigma_y, 0, mix_id)

            x_samp = np.random.normal(0,1,1)
            x_samp = torch.Tensor(x_samp).view(1,1,-1)
            
            y_samp = np.random.normal(0,1,1)
            y_samp = torch.Tensor(y_samp).view(1,1,-1)

            x = mu_x + sigma_x*x_samp
            y = mu_y + sigma_y*(y_samp*(1-ro.pow(2)).pow(1/2) + ro*x_samp)
            end_stroke = np.random.binomial(1,e.data.flatten())
            # end_stroke = torch.Tensor(end_stroke)
            # import pdb;pdb.set_trace()

            new_seq[seq_time] = [end_stroke, list(x.data.numpy().flatten())[0], list(y.data.numpy().flatten())[0]]
            # import pdb;pdb.set_trace()
        # return new_seq
        un_normalized_output = utils.un_normalize_data([new_seq], train_mean, train_std)
        utils.plot_stroke(un_normalized_output, "/content/gen.png")






