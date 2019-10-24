import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Model(nn.Module):
	"""docstring for Model"""
	def __init__(self, inp_dim, hidden_units, out_dim, num_layers):
		super(Model, self).__init__()

		self.num_layers = num_layers
		self.inp_dim = inp_dim
		self.out_dim = out_dim
		self.layer1 = nn.LSTM(inp_dim, hidden_units, batch_first = True)
		self.layers = []
		for layer_num in range(1, num_layers):
			self.layers.append(nn.LSTM(inp_dim + hidden_units, hidden_units, batch_first = True))
		self.layers = nn.ModuleList(self.layers)
		# self.lstm = nn.Sequential(*layers)
		
		self.l1 = nn.Linear(num_layers * hidden_units, out_dim)

	def forward(self, X, lens):
		# import pdb; pdb.set_trace()

		lens, perm_idx = lens.sort(0, descending=True)
		X = X[perm_idx]
		X = X.float()
		# print(X.type())
		# import pdb; pdb.set_trace()
		packed_input = pack_padded_sequence(X, lens.cpu().numpy(), batch_first=True)

		# import pdb; pdb.set_trace()
		out, _ = self.layer1(packed_input)
		# import pdb; pdb.set_trace()
		output, input_sizes = pad_packed_sequence(out, batch_first=True)

		outs = [output]

		# import pdb; pdb.set_trace()
		
		for layer in self.layers:
			x_out = torch.cat((X,output), dim=2)
			# import pdb; pdb.set_trace()
			packed_input = pack_padded_sequence(x_out, lens.cpu().numpy(), batch_first=True)
			# import pdb; pdb.set_trace()
			out, _ = layer(packed_input)
			# import pdb; pdb.set_trace()
			output, input_sizes = pad_packed_sequence(out, batch_first=True)
			# import pdb; pdb.set_trace()
			outs.append(output)
		# import pdb; pdb.set_trace()
		all_outs = torch.cat(outs, dim = 2)
		# import pdb; pdb.set_trace()
		pred = self.l1(all_outs)
		# import pdb; pdb.set_trace()
		# print(pred.shape)

		return pred


# if __name__ == '__main__':
# 	model = Model(3, 214, 4, 3)
# 	inp = torch.randn((2,3,3))
# 	model(inp)
