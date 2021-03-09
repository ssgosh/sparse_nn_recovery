import pathlib
import pickle
import sys

import torch
from torchvision.transforms import transforms

sys.path.append(".")

import argparse

from torch.utils.data import DataLoader, Dataset, Subset

from datasets.dataset_helper_factory import DatasetHelperFactory

parser = argparse.ArgumentParser(
    description='Splits train dataset into multiple train and validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, default='mnist',
                    metavar='D',
                    help='Which dataset to split (e.g. MNIST)')
config = parser.parse_args()
transform = transforms.ToTensor()
dname = config.dataset
ds = DatasetHelperFactory.get(dname).get_dataset(which='train', transform=transform)
#ds = DatasetHelperFactory.get(config.dataset).get_dataset(train=True, transform=None)


def save(filename, dataset):
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def split(N, name):
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
    save(f'./data/{name}_A/idx/train.p', train_A)
    save(f'./data/{name}_A/idx/test.p', valid_A)
    save(f'./data/{name}_B/idx/train.p', train_B)
    save(f'./data/{name}_B/idx/test.p', valid_B)

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
    which = "train" if train else "test"
    loaded = (DatasetHelperFactory.get_new(name).get_dataset(which, transform))

    # First print stats
    get_dataset_stats(f'Saved {name}, {which}', saved)
    get_dataset_stats(f'Loaded {name}, {which}', loaded)
    saved_loader = DataLoader(saved)
    loaded_loader = DataLoader(loaded)
    for i, ((saved_img, saved_tg), (loaded_img, loaded_tgt)) in enumerate(zip(saved_loader, loaded_loader)):
        assert torch.all(saved_img == loaded_img).item(), f"Did not match {name}, {i}"
    print("All Good!!!")


# Split and return indices
train_A, train_B, valid_A, valid_B = split(len(ds), dname)

get_dataset_stats(f'Full {dname}', ds)

compare_with_loaded(train_A, f'{dname}_A', train=True)
compare_with_loaded(train_B, f'{dname}_B', train=True)
compare_with_loaded(valid_A, f'{dname}_A', train=False)
compare_with_loaded(valid_B, f'{dname}_B', train=False)
