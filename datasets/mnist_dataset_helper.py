from torchvision import datasets
from torchvision.transforms import transforms

from datasets.dataset_helper import DatasetHelper
from datasets.non_sparse_normalization_mixin import NonSparseNormalizationMixin
from models.mnist_model import ExampleCNNNet
from utils import mnist_helper


class MNISTdatasetHelper(DatasetHelper, ):
    def __init__(self, name, subset, non_sparse):
        super().__init__(name, subset, non_sparse)

        # These implement the non-sparse normalization mixin
        self.train_transforms = [transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),]
        self.test_transforms = []
        self.usual_mean = 0.1307
        self.usual_std = 0.3081
        self.non_sparse_mean = 0.4032246160182823
        self.non_sparse_std = 0.23552181428064123
        self.constant_pixel_val = 0.3

    def get_dataset_(self, path, which, transform):
        return datasets.MNIST(path, train=(which == 'train'), transform=transform)

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

