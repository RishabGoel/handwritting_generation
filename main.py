import torch
from torch.utils import data
from utils.helper import *
from dataloader import Dataset


def main():
    train_x, train_y, test_x, test_y = get_data('data\\strokes.npy')
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    training_set = Dataset(train_x, train_y)
    training_generator = data.DataLoader(training_set, **params)

main()
