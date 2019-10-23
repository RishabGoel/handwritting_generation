import numpy as np

def create_splits(path_to_data, create_valid = False):
	
	data = np.load(path_to_data, allow_pickle=True, encoding = 'latin1')
	
	train = data[ : int(0.8 * data.shape[0])]
	
	if create_valid:
	
		valid = data[int(0.8 * data.shape[0]) : int(0.9 * data.shape[0])]
		test = data[int(0.9 * data.shape[0]) : ]
		
		assert(train.shape[0] + valid.shape[0] + test.shape[0] == data.shape[0])

	else:
	
		test = data[int(0.8 * data.shape[0]) : ]

		assert(train.shape[0] + test.shape[0] == data.shape[0])

	return train, test 

def create_xy(data):
	x = []
	y = []
	
	for data_idx in range(data.shape[0]):
		x.append(data[data_idx][:-1])
		y.append(data[data_idx][1:])

		assert(x[-1].shape==y[-1].shape)
	# print(x[0].shape,)
	x = np.array(x)
	y = np.array(y)

	assert(x.shape == y.shape)

	return x, y

def get_stats(data):
	x = []
	y =  []
	for i in range(data.shape[0]):
		x.append(data[i][:,1])
		y.append(data[i][:,2])

	assert(len(x)==data.shape[0])
	assert(len(y)==data.shape[0])

	print(len(x))
	
	x = np.concatenate(x)
	y = np.concatenate(y)

	assert(x.shape==y.shape)
	
	print(x.shape)

	# import pdb;pdb.set_trace()
	
	return [np.mean(x), np.mean(y)], [np.std(x), np.std(y)]

def normalize_data(data, mean, std):
	norm_data = []
	
	for i in range(data.shape[0]):
		tmp = np.copy(data[i])
		# import pdb; pdb.set_trace()
		tmp[:,1] -= mean[0]
		tmp[:,1] /= std[0]

		tmp[:,2] -= mean[1]
		tmp[:,2] /= std[1]
		# import pdb; pdb.set_trace()
		norm_data.append(tmp)
		# import pdb; pdb.set_trace()
	
	return np.array(norm_data)

def un_normalize_data(data, mean, std):
	norm_data = []
	
	for i in range(data.shape[0]):
		tmp = np.copy(data[i])
		tmp[:, 1] *= std[0]
		tmp[:, 1] += mean[0]

		tmp[:, 2] *= std[1]
		tmp[:, 2] += mean[1]

		norm_data.append(tmp)

	return np.array(norm_data)



def get_data(path_to_data = '..\\data\\strokes.npy'):
	
	train, test = create_splits(path_to_data)
	train_mean, train_std = get_stats(np.copy(train))

	train_norm, test_norm = normalize_data(train, train_mean, train_std),normalize_data(test, train_mean, train_std) 

	train_un_norm, test_un_norm = un_normalize_data(train_norm, train_mean, train_std),un_normalize_data(test_norm, train_mean, train_std)
	# import pdb; pdb.set_trace()
	# assert(np.array_equal(train, train_un_norm))
	# import pdb; pdb.set_trace()
	# assert(np.array_equal(test, test_un_norm))
	# import pdb; pdb.set_trace()
	train_x, train_y = create_xy(train_norm)
	test_x, test_y = create_xy(test_norm)
	print(train.shape, test.shape, train_x.shape, train_y.shape, test_x.shape, test_y.shape, train[0].shape, train_x[0].shape)
	return train_x, train_y, test_x, test_y, train_mean, train_std