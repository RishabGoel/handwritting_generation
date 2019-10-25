import argparse
import torch

from utils.helper import *
from trainer import Trainer

parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')

parser.add_argument('-bs', type=int, default='64',
                    help='Batch size to be used for training')
parser.add_argument('-hidden_units', type=int, default=400,
                    help='size of hidden layers of lstm')
parser.add_argument('-num_layers', type=int, default=2,
                    help='numberof lstm layers to be used')
parser.add_argument('-inp_dim', type=int, default=3,
                    help='input dimension to lstm')
parser.add_argument('-out_dim', type=int, default=4,
                    help='final output dimension')
parser.add_argument('-epochs', type=int, default=4,
                    help='number of epochs to run the model for')
parser.add_argument('-lr', type=float, default=0.001,
                    help='learning rate to be used for training the model')
parser.add_argument('-seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('-num_mixtures', type=int, default=3,
                    help='Number of mixtures to be use')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main(args):
    print("in main")
    train_x, train_y, test_x, test_y, train_mean, train_std = get_data('data\\strokes.npy')
    print(args)
    # Train Model
    trainer = Trainer(args)
    trainer.fit(train_x, train_y)
    return

main(args)
