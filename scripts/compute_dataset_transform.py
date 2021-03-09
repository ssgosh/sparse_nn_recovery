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

parser.add_argument('--dataset', type=str, default='mnist',
                    metavar='D',
                    help='Which dataset to split (e.g. MNIST)')
parser.add_argument('--non-sparse', action='store_true', dest='non_sparse', default=False,
                    help='Whether to use non-sparse version')
config = parser.parse_args()
dh = DatasetHelperFactory.get(config.dataset, non_sparse=config.non_sparse)

train = dh.get_dataset(which='train', transform='without_transform')
test = dh.get_dataset(which='test', transform='without_transform')

total = 0.
total_sq = 0.
num = 0
for ds in [train]:
    dl = DataLoader(ds, batch_size=1000)
    for images, targets in dl:
        total += torch.sum(images).item()
        total_sq += torch.sum(images * images).item()
        num += torch.flatten(images).shape[0]


mean = total / num
mean_sq = total_sq / num
std = math.sqrt(mean_sq - mean**2)

print(f'mean = {mean}, std = {std}')
# for ds in [train, test]:
#     dl = DataLoader(ds, batch_size=1000)
#     for images, targets in dl:
#         total += torch.sum(images).item()
#         num += torch.flatten(images).shape[0]
