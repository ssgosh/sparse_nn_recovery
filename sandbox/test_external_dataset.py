import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.dataset_helper_factory import DatasetHelperFactory

def get_dataset_stats(name, dataset : Dataset):
    print("Dataset :", name)
    loader = DataLoader(dataset, batch_size=len(dataset))
    img, tgt = next(iter(loader))
    print(img.shape, tgt.shape)
    stats = torch.bincount(tgt)
    print("\n".join([f"{i} : {n}" for i, n in enumerate(stats)]))

ds = DatasetHelperFactory.get_new('external_B').get_dataset(which='test')
get_dataset_stats(f'external_B_test', ds)

ds = DatasetHelperFactory.get_new('external_B').get_dataset(which='valid')
get_dataset_stats(f'external_B_valid', ds)
