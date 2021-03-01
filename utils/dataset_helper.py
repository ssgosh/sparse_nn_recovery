from abc import ABC, abstractmethod

from models.mnist_model import ExampleCNNNet
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
                classobj.dataset = MNISTdatasetHelper()
                return classobj.dataset
            elif dataset_name == 'cifar':
                classobj.dataset = CIFARDatasetHelper()
                return classobj.dataset
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

    def setup_config(self, config):
        zero, one = self.get_transformed_zero_one()
        config.image_zero = zero
        config.image_one = one
        self.update_config(config)

    @abstractmethod
    def get_num_classes(self):
        pass

    # No need to be abstractmethod
    def get_num_real_fake_classes(self):
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
    def __init__(self):
        super().__init__()

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


class CIFARDatasetHelper(DatasetHelper):
    def __init__(self):
        super().__init__()

    def get_transformed_zero_one(self):
        raise NotImplementedError()

