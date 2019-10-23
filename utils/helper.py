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
	pass

def normalize_data():
	pass


train, test = create_splits('..\\data\\strokes.npy')

train_x, train_y = create_xy(train)
test_x, test_y = create_xy(test)
print(train.shape, test.shape, train_x.shape, train_y.shape, test_x.shape, test_y.shape, train[0].shape, train_x[0].shape)