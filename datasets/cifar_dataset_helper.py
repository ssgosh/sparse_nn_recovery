from icontract import require
from torchvision import datasets
from torchvision.transforms import transforms

from datasets.dataset_helper import DatasetHelper


class CIFARDatasetHelper(DatasetHelper):
    @require(lambda non_sparse: not non_sparse)
    def __init__(self, name, subset, non_sparse):
        super().__init__(name, subset, non_sparse)

        # These implement the non-sparse normalization mixin
        self.train_transforms = [
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),
        ]
        self.test_transforms = []
        self.usual_mean = (0.4914, 0.4822, 0.4465)
        self.usual_std = (0.2023, 0.1994, 0.2010)
        self.non_sparse_mean = {}
        self.non_sparse_std = {}
        self.constant_pixel_val = 0.3

        # Further needed by the non-sparse mixin
        # These are transformed values of zero and one pixel values
        self.usual_zero = [-2.429065704345703, -2.418254852294922, -2.22139310836792]
        self.usual_one = [2.5140879154205322, 2.596790313720703, 2.7537312507629395]

        self.non_sparse_zero = {}
        self.non_sparse_one = {}

    def get_dataset_(self, path, which, transform):
        return datasets.CIFAR10(path, train=(which == 'train'), transform=transform)

    def get_num_classes(self):
        return 10

    def get_each_entry_shape(self):
        return (3, 28, 28)

    def get_model(self, model_mode, device):
        model = None
        if model_mode == 'fake-classes':
            raise ValueError(f"Model mode {model_mode} not supported for CIFAR10")
        elif model_mode == 'max-entropy':
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid model mode {model_mode}")
        # model = MLPNet().to(device)
        # model = MLPNet3Layer(num_classes=20).to(device)
        # model = MaxNormMLP(num_classes=20).to(device)
        return model

    def update_config(self, config):
        config.model_classname = 'Resnet18'

