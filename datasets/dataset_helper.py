import pickle
from abc import ABC, abstractmethod

from torch.utils.data import Dataset, Subset
from torchvision import datasets

from models.mnist_model import ExampleCNNNet
from utils import mnist_helper


# Abstraction for Dataset-specific functionality, such as transformations,
# values of transformed zero and one pixel values. Also provides a singleton for the datasethelper
# Used in the main scripts.
class DatasetHelper(ABC):
    dataset = None

    @classmethod
    def get(classobj, dataset_name : str = None):
        """
        Singleton method

        :param dataset_name:
        :return:
        """
        if classobj.dataset is None:
            classobj.dataset = classobj.get_new(dataset_name)
            return classobj.dataset
        else:
            assert dataset_name is None
            return classobj.dataset

    @classmethod
    def get_new(classobj, cased_dataset_name):
        """
        Factory method to get new datasets

        :param dataset_name:
        :return:
        """
        assert cased_dataset_name
        dataset_name = cased_dataset_name.lower()
        subset = dataset_name not in ['mnist', 'cifar']
        if 'mnist' in dataset_name:
            return MNISTdatasetHelper(name=cased_dataset_name, subset=subset)
        elif 'cifar' in dataset_name:
            return CIFARDatasetHelper(name=cased_dataset_name, subset=subset)
        else:
            raise ValueError("Invalid dataset name: %s" % dataset_name)

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


class MNISTdatasetHelper(DatasetHelper):
    def __init__(self, name, subset):
        super().__init__(name, subset)

    def get_dataset_(self, path, train, transform):
        return datasets.MNIST(path, train=train, transform=transform)

    def get_transformed_zero_one(self):
        return mnist_helper.compute_mnist_transform_low_high()

    def get_num_classes(self):
        return 10

    def get_each_entry_shape(self):
        return (1, 28, 28)

    def get_model(self, model_mode, device):
        if model_mode == 'fake-classes': model = ExampleCNNNet(20).to(device)
        elif model_mode == 'max-entropy': model = ExampleCNNNet(10).to(device)
        else: raise ValueError(f"Invalid mode mode {model_mode}")
        # model = MLPNet().to(device)
        # model = MLPNet3Layer(num_classes=20).to(device)
        # model = MaxNormMLP(num_classes=20).to(device)
        return model

    def update_config(self, config):
        config.model_classname = 'ExampleCNNNet'

#class MNISTSubsetDatasetHelper(MNISTdatasetHelper):
#    def __init__(self, ):
#        super(MNISTSubsetDatasetHelper, self).__init__()

class CIFARDatasetHelper(DatasetHelper):
    def __init__(self, name, subset):
        super().__init__(name, subset)

    def get_transformed_zero_one(self):
        raise NotImplementedError()

