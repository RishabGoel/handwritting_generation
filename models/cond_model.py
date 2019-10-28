import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Model(nn.Module):
	"""docstring for Model"""
	def __init__(self, inp_dim, hidden_units, out_dim, num_layers, num_mixtures = 20, chars = 54, bias = 3, window_mixtures = 10):
		super(Model, self).__init__()

		self.num_layers = num_layers
		self.inp_dim = inp_dim
		self.out_dim = out_dim
		self.hidden_units = hidden_units
		self.bias = bias
		self.device = 'cuda'
		self.chars = chars + 1
		self.num_mixtures = num_mixtures
		self.window_mixtures = window_mixtures
		self.layer1 = nn.LSTM(inp_dim, hidden_units, batch_first = True)
		self.layer2 = nn.LSTM(inp_dim + hidden_units + self.chars, hidden_units)
		self.create_window_vars()
		self.create_mdn_vars()
		
		self.tan = nn.Tanh()
		self.softmax = nn.Softmax(dim = 2)
		self.sigmoid = nn.Sigmoid()


	def create_mdn_vars(self):
		self.mu = nn.Linear(self.num_layers * self.hidden_units, 2 * self.num_mixtures)
		self.sigma = nn.Linear(self.num_layers * self.hidden_units, 2 * self.num_mixtures)
		self.pi = nn.Linear(self.num_layers * self.hidden_units, self.num_mixtures)
		self.ro = nn.Linear(self.num_layers * self.hidden_units, self.num_mixtures)

		self.e = nn.Linear(self.num_layers * self.hidden_units, 1)


	def create_window_vars(self):
		self.a = nn.Linear(self.hidden_units,self.window_mixtures)
		self.b = nn.Linear(self.hidden_units,self.window_mixtures)
		self.k = nn.Linear(self.hidden_units,self.window_mixtures)
	
	def init_hidden_cell_state(self, bs):
		hc_lst = []
		
		for i in range(self.num_layers):
			hc_lst.append((torch.zeros(1,bs,self.hidden_units).to(self.device), torch.zeros(1,bs,self.hidden_units).to(self.device)))

		return hc_lst

	def window_forward(self, hidden_outs, context, prev_k = None):
		a_out = torch.exp(self.a(hidden_outs)).unsqueeze(-1)
		b_out = torch.exp(self.b(hidden_outs)).unsqueeze(-1)
		k_out = torch.exp(self.k(hidden_outs)).unsqueeze(-1)
		
		if prev_k is not None:
			k_out += prev_k
		else:
			k_out = torch.cumsum(k_out, dim = 1)
		
		k_out = torch.exp(k_out).unsqueeze(-1)
		u = torch.autograd.Variable(torch.arange(0, context.shape[1])).view(1,1,1,-1)
		phi_k_u = torch.sum(a_out * torch.exp(- b_out * (k_out - u)), dim = 2)
		w = torch.matmul(phi_k_u)
		return k_out, phi_k_u, w

	def forward(self, X, context, hc_lst = None, prev_k = None):
		# import pdb; pdb.set_trace()

		if hc_lst is None:
			hc_lst = self.init_hidden_cell_state(X.shape[0])
		
		outs1, hc1 = self.layer1(X, hc_lst[0])

		final_hc_lst = [hc1]

		k, phi, w = self.window_forward(outs1, context, prev_k)

		inp1 = torch.cat((outs1, X, w), dim = -1)

		outs2, hc2 = self.layer2(inp1, hc_lst[1])
		final_hc_lst.append(hc2)

		all_outs = torch.cat((outs1, outs2), dim = -1)
		# import pdb; pdb.set_trace()
		e = self.sigmoid(self.e(all_outs))
		mu = self.mu(all_outs)
		sigma = torch.exp(self.sigma(all_outs))
		ro = self.tan(self.ro(all_outs))
		pi = self.softmax(self.pi(all_outs))
		# import pdb; pdb.set_trace()


		return e,ro,pi,mu,sigma, final_hc_lst, k


# if __name__ == '__main__':
# 	model = Model(3, 214, 4, 3)
# 	inp = torch.randn((2,3,3))
# 	model(inp)
