import pathlib
import pickle
import sys

import torch
import numpy as np
from torchvision.transforms import transforms

sys.path.append(".")

import argparse

from torch.utils.data import random_split, DataLoader, Dataset

from utils.dataset_helper import DatasetHelper

parser = argparse.ArgumentParser(
    description='Splits train dataset into multiple train and validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, default='mnist',
                    metavar='D',
                    help='Which dataset to split (e.g. MNIST)')
config = parser.parse_args()
#dataset = DatasetHelper.get(config.dataset).get_dataset(train=True, transform=transforms.ToTensor())
ds = DatasetHelper.get(config.dataset).get_dataset(train=True, transform=None)


def save(filename, dataset):
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def split(dataset):
    m = 25_000
    n = 5_000
    N = len(dataset)
    idx = torch.randperm(N)
    #train_A, train_B, valid_A, valid_B = random_split(dataset, (m, m, n, n))
    # Only save indices of the splits. Subsets of MNIST will be created while loading, on the fly
    train_A = idx[0:m]
    train_B = idx[m:2*m]
    valid_A = idx[2*m : 2*m + n]
    valid_B = idx[2*m + n : 2*m + 2*n]
    #random_split(dataset, (m, m, n, n))
    print(type(train_A))
    print(len(train_A), len(train_B), len(valid_A), len(valid_B))
    save('./data/MNIST_A/idx/train.p', train_A)
    save('./data/MNIST_A/idx/test.p', valid_A)
    save('./data/MNIST_B/idx/train.p', train_B)
    save('./data/MNIST_B/idx/test.p', valid_B)

def get_dataset_stats(name, dataset : Dataset):
    #transform = transforms.ToTensor()
    #dataset.transform = transform
    #dataset.target_transform = transform

    #dataset = Dataset(dataset, transform= )
    loader = DataLoader(dataset, batch_size=len(dataset))
    img, tgt = next(iter(loader))
    print(img.shape, tgt.shape)
    stats = torch.bincount(tgt)
    print("\n".join([f"{i} : {n}" for i, n in enumerate(stats)]))

split(ds)
#get_dataset_stats('MNIST', dataset)
#get_dataset_stats('train_A', train_A)
#get_dataset_stats('valid_A', valid_A)
#get_dataset_stats('train_B', train_B)
#get_dataset_stats('valid_B', valid_B)
