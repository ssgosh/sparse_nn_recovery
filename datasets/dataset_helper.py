import pickle
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from datasets.non_sparse_normalization_mixin import NonSparseNormalizationMixin


# Abstraction for Dataset-specific functionality, such as transformations,
# values of transformed zero and one pixel values. Also provides a singleton for the datasethelper
# Used in the main scripts.
class DatasetHelper(ABC, NonSparseNormalizationMixin):

    def __init__(self, name, subset, non_sparse):
        self.name = name
        self.subset = subset
        self.non_sparse = non_sparse

    def get_dataset(self, which='train', transform=None) -> Dataset:
        """
        Return torch.utils.data.Dataset object for train/test/valid split with the given transform

        :param which:
        :param transform:
        :return:
        """
        train = (which == 'train')
        if transform == 'train' : transform = self.get_train_transform()
        elif transform == 'test' : transform = self.get_test_transform()
        elif transform == 'without_transform' : transform = self.get_without_transform()
        path = './data'
        if not self.subset : return self.get_dataset_(path, which, transform)

        # train/test splits were created from original train. Hence train=True in the following call.
        full_train_data = self.get_dataset_(path, which='train', transform=transform)
        fname = 'train' if train else 'test'
        path = f'./data/{self.name}/idx/{fname}.p'
        with open(path, 'rb') as f:
            idx = pickle.load(f)
        return Subset(full_train_data, idx)

    @abstractmethod
    def get_dataset_(self, path, which, transform):
        pass

    def get_transformed_zero_one(self):
        return self.get_transformed_zero_one_mixin()

    def setup_config(self, config):
        zero, one = self.get_transformed_zero_one()
        config.image_zero = zero
        config.image_one = one
        self.update_config(config)

    # XXX: Rename method to get_num_real_classes and update usage.
    @abstractmethod
    def get_num_classes(self):
        pass

    # No need to be abstractmethod
    def get_num_real_fake_classes(self):
        # XXX: Change this to get_num_real_classes()
        return 2 * self.get_num_classes()

    # XXX: This should use adversarial_classification_mode and either return only real or real + fake classesj
    def get_num_total_classes(self):
        return self.get_num_classes()

    @abstractmethod
    def get_each_entry_shape(self):
        pass

    @abstractmethod
    def get_model(self, model_mode, device):
        pass

    @abstractmethod
    def update_config(self, config):
        pass

    def get_train_transform(self):
        return self.get_train_transform_()

    def get_test_transform(self):
        return self.get_test_transform_()

    def get_without_transform(self):
        return self.get_without_transform_()

    def compute_transform_low_high(self):
        """
        Computes the transformed zero and one pixel values.

        Should typically not be used. Use get_transformed_zero_one() instead, which uses hard-coded values.
        Those hard-coded values are computed using this function in the script scripts/compute_dataset_stats.py
        """
        mean, std = self.get_mean_std()
        transform = transforms.Normalize(mean, std)
        low = torch.zeros(1, 1, 1)
        high = low + 1
        print(torch.sum(low).item(), torch.sum(high).item())
        transformed_low = transform(low).item()
        transformed_high = transform(high).item()
        print(transformed_low, transformed_high)
        return transformed_low, transformed_high

    def get_mean_std(self):
        return self.get_mean_std_mixin()
