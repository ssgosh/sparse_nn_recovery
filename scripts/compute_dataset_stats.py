"""Computes the mean and std of an entire dataset"""
import sys
sys.path.insert(0, ".")

import argparse

import torch
from torch.utils.data import DataLoader

from datasets.dataset_helper_factory import DatasetHelperFactory
from utils.image_processor import save_grid_of_images

parser = argparse.ArgumentParser(
    description='Computes the mean and std of an entire dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, default='MNIST',
                    metavar='D',
                    help='Dataset to give stats on (e.g. MNIST)')
stats_opts = ['mean_std', 'zero_one', 'sparsity', 'view-images']
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

    channels = dh.get_each_entry_shape()[0]
    total = torch.zeros(channels)
    total_sq = torch.zeros(channels)
    num = 0
    for ds in datasets:
        dl = DataLoader(ds, batch_size=1000)
        for images, targets in dl:
            total += torch.sum(images, (0, 2, 3))
            total_sq += torch.sum(images * images, (0, 2, 3))
            num += images.shape[0] * images.shape[2] * images.shape[3]

    mean = total / num
    mean_sq = total_sq / num
    std = torch.sqrt(mean_sq - mean**2)

    print(f'mean = {mean.tolist()}, std = {std.tolist()}')

elif config.stats == 'zero_one':
    zero, one = dh.compute_transform_low_high()
    print(f'zero = {zero}, one = {one}')

elif config.stats == 'sparsity':
    zeros = torch.zeros_like(dh.get_zero_correct_dims())
    train = dh.get_dataset(which='train', transform='without_transform')
    dl = DataLoader(train, batch_size=len(train))
    #idx = torch.randperm(torch.arange(len(train)))
    images, targets = next(iter(dl))
    n = len(zeros.shape)
    per_channel_sparsity = torch.sum(images > zeros, dim=list(range(2, n))).float()
    sparsity = torch.sum(images > zeros, dim=list(range(1, n))).float()

    #torch.mean(, dim=0)

    def print_hist(hist, bins, min, max):
        step = (max - min) / 10
        for i in range(bins):
            print(f'{min + i*step} - {min + (i+1)*step} : {hist[i]}')

    def print_sparsity(msg, sparsity):
        mean = torch.mean(sparsity, dim=0).numpy()
        median = torch.median(sparsity, dim=0).values.numpy()
        std = torch.std(sparsity, dim=0).numpy()
        max = torch.max(sparsity, dim=0).values.numpy()
        min = torch.min(sparsity, dim=0).values.numpy()
        hist = torch.histc(sparsity, bins=10, min=0, max=500)
        print(msg, f'mean = {mean}, median = {median}, std = {std}, min = {min}, max = {max}')
        print_hist(hist, 10, 0, 500)

    print_sparsity('Total sparisty', sparsity)
    print_sparsity('Per-channel sparsity: ', per_channel_sparsity)
elif config.stats == 'view-images':
    # Sample 100 images from the dataset and view them
    if config.dataset.lower() == 'external_b' or config.dataset.lower() == 'external_b_non_sparse':
        ds = dh.get_dataset(which='valid', transform=None)
        dl = DataLoader(ds, batch_size=len(ds), shuffle=True)
        images, targets = next(iter(dl))
        save_grid_of_images(f'{config.dataset}_samples.png', images, targets, dh, sf=10.0)

    else:
        assert False
