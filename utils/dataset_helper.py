from abc import ABC, abstractmethod
from utils import mnist_helper

# Abstraction for Dataset-specific functionality, such as transformations,
# values of transformed zero and one pixel values. Also provides a singleton for the datasethelper
# Used in the main scripts.
class DatasetHelper(ABC):
    dataset = None

    @classmethod
    def get_dataset(classobj, dataset_name : str = None):
        if classobj.dataset is None:
            assert dataset_name is not None
            if dataset_name == 'mnist':
                return MNISTdatasetHelper()
            elif dataset_name == 'cifar':
                return CIFARDatasetHelper()
            else:
                raise ValueError("Invalid dataset name: %s" % dataset_name)
        else:
            assert dataset_name is None
            return classobj.dataset

    def __init__(self):
        pass

    @abstractmethod
    def get_transformed_zero_one(self):
        pass


class MNISTdatasetHelper(DatasetHelper):
    def __init__(self):
        super().__init__()

    def get_transformed_zero_one(self):
        return mnist_helper.compute_mnist_transform_low_high()


class CIFARDatasetHelper(DatasetHelper):
    def __init__(self):
        super().__init__()

    def get_transformed_zero_one(self):
        raise NotImplementedError()
