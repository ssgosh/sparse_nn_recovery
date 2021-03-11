"""Computes the mean and std of an entire dataset"""
import sys
sys.path.insert(0, ".")
import argparse
import math

import torch
from torch.utils.data import DataLoader

from datasets.dataset_helper_factory import DatasetHelperFactory

parser = argparse.ArgumentParser(
    description='Computes the mean and std of an entire dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, default='MNIST',
                    metavar='D',
                    help='Dataset to give stats on (e.g. MNIST)')
stats_opts = ['mean_std', 'zero_one']
parser.add_argument('--stats', type=str, required=True, choices=stats_opts,
                    help='Which stats')
parser.add_argument('--which', type=str, default='train',
                    help='Which data type to use (train/test/all)')
parser.add_argument('--non-sparse', action='store_true', dest='non_sparse', default=False,
                    help='Whether to use non-sparse version')
transformations = ['without_transform', 'train', 'test']
parser.add_argument('--transform', type=str, default='without_transform',
                    choices = transformations,
                    help='Which transformation to apply')
config = parser.parse_args()
dh = DatasetHelperFactory.get(config.dataset, non_sparse=config.non_sparse)

if config.stats == 'mean_std':
    train = dh.get_dataset(which='train', transform=config.transform)
    test = dh.get_dataset(which='test', transform=config.transform)

    # Switch/case in Python
    choices = {
        'train': [train],
        'test': [test],
        'all' : [train, test]
    }
    datasets = choices[config.which]

    total = 0.
    total_sq = 0.
    num = 0
    for ds in datasets:
        dl = DataLoader(ds, batch_size=1000)
        for images, targets in dl:
            total += torch.sum(images).item()
            total_sq += torch.sum(images * images).item()
            num += torch.flatten(images).shape[0]


    mean = total / num
    mean_sq = total_sq / num
    std = math.sqrt(mean_sq - mean**2)

    print(f'mean = {mean}, std = {std}')
elif config.stats == 'zero_one':
    zero, one = dh.compute_transform_low_high()
    print(f'zero = {zero}, one = {one}')
