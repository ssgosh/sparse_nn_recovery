import pathlib
import pickle
import sys

import torch
from torchvision.transforms import transforms

sys.path.append(".")

import argparse

from torch.utils.data import DataLoader, Dataset, Subset

from datasets.dataset_helper import DatasetHelper

parser = argparse.ArgumentParser(
    description='Splits train dataset into multiple train and validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, default='mnist',
                    metavar='D',
                    help='Which dataset to split (e.g. MNIST)')
config = parser.parse_args()
transform = transforms.ToTensor()
ds = DatasetHelper.get(config.dataset).get_dataset(train=True, transform=transform)
#ds = DatasetHelper.get(config.dataset).get_dataset(train=True, transform=None)


def save(filename, dataset):
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def split(N):
    m = 25_000
    n = 5_000
    #N = len(dataset)
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

    return train_A, train_B, valid_A, valid_B

def get_dataset_stats(name, dataset : Dataset):
    print("Dataset :", name)
    loader = DataLoader(dataset, batch_size=len(dataset))
    img, tgt = next(iter(loader))
    print(img.shape, tgt.shape)
    stats = torch.bincount(tgt)
    print("\n".join([f"{i} : {n}" for i, n in enumerate(stats)]))

def compare_with_loaded(idx, name, train):
    global transform, ds
    saved = (Subset(ds, idx))
    loaded = (DatasetHelper.get_new(name).get_dataset(train, transform))

    # First print stats
    get_dataset_stats(f'Saved {name}, {"train" if train else "test"}', saved)
    get_dataset_stats(f'Loaded {name}, {"train" if train else "test"}', loaded)
    saved_loader = DataLoader(saved)
    loaded_loader = DataLoader(loaded)
    for i, ((saved_img, saved_tg), (loaded_img, loaded_tgt)) in enumerate(zip(saved_loader, loaded_loader)):
        assert torch.all(saved_img == loaded_img).item(), f"Did not match {name}, {i}"
    print("All Good!!!")


# Split and return indices
train_A, train_B, valid_A, valid_B = split(len(ds))

get_dataset_stats('Full MNIST', ds)

compare_with_loaded(train_A, 'MNIST_A', train=True)
compare_with_loaded(train_B, 'MNIST_B', train=True)
compare_with_loaded(valid_A, 'MNIST_A', train=False)
compare_with_loaded(valid_B, 'MNIST_B', train=False)
