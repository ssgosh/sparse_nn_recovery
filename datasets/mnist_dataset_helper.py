from torchvision import datasets

from datasets.dataset_helper import DatasetHelper
from models.mnist_model import ExampleCNNNet
from utils import mnist_helper


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