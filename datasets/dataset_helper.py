import pickle
from abc import ABC, abstractmethod

from torch.utils.data import Dataset, Subset


# Abstraction for Dataset-specific functionality, such as transformations,
# values of transformed zero and one pixel values. Also provides a singleton for the datasethelper
# Used in the main scripts.
class DatasetHelper(ABC):

    def __init__(self, name, subset):
        self.name = name
        self.subset = subset

    def get_dataset(self, train=True, transform=None) -> Dataset:
        if transform == 'train' : transform = self.get_train_transform()
        elif transform == 'test' : transform = self.get_test_transform()
        path = './data'
        if not self.subset : return self.get_dataset_(path, train, transform)

        # train/test splits were created from original train. Hence train=True in the following call.
        full_train_data = self.get_dataset_(path, train=True, transform=transform)
        fname = 'train' if train else 'test'
        path = f'./data/{self.name}/idx/{fname}.p'
        with open(path, 'rb') as f:
            idx = pickle.load(f)
        return Subset(full_train_data, idx)

    @abstractmethod
    def get_dataset_(self, path, train, transform):
        pass

    #@abstractmethod
    def get_train_transform(self):
        raise NotImplementedError("Not yet implemented")

    #@abstractmethod
    def get_test_transform(self):
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def get_transformed_zero_one(self):
        pass

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

    @abstractmethod
    def get_each_entry_shape(self):
        pass

    @abstractmethod
    def get_model(self, model_mode, device):
        pass

    @abstractmethod
    def update_config(self, config):
        pass


