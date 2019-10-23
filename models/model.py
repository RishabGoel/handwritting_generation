import torch
import torch.nn as nn

class Model(nn.Module):
	"""docstring for Model"""
	def __init__(self, inp_features, hidden_units, out_features, num_layers):
		super(Model, self).__init__()

		self.num_layers = num_layers
		self.inp_dim = inp_dim
		self.out_dim = out_dim

		self.layers = [nn.LSTM(inp_dim, hidden_units, batch_first = True)]
		for layer_num in range(1, num_layers):
			self.layers.append(nn.LSTM(inp_dim + hidden_units, hidden_units, batch_first = True))

		# self.lstm = nn.Sequential(*layers)
		self.l1 = nn.Linear(len(num_layers) * hidden_units, out_dim)

	def forward(self, X):

		out, self.hidden = self.layers[0]
		# for
		
		print(lstm_out.shape)
