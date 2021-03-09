"""Computes the mean and std of an entire dataset"""
import argparse

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
parser.add_argument('--non-sparse', actions='store_true', default=False,
                    help='Whether to use non-sparse version')
config = parser.parse_args()
dh = DatasetHelperFactory.get(config.dataset, non_sparse=config.non_sparse)

train = dh.get_dataset(which='train', transform='without_transform')
test = dh.get_dataset(which='test', transform='without_transform')

total = 0.
num = 0
for ds in [train, test]:
    dl = DataLoader(ds, batch_size=1000)
    for images, targets in dl:
        total += torch.sum(images)
        num += torch.flatten(images).shape[0]

mean = total / num