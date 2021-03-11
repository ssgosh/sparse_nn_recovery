from icontract import require
from torchvision.transforms import transforms

from datasets.dataset_helper import DatasetHelper


class CIFARdatasetHelper(DatasetHelper):
    @require(lambda non_sparse: not non_sparse)
    def __init__(self, name, subset, non_sparse):
        super().__init__(name, subset, non_sparse)

        # These implement the non-sparse normalization mixin
        self.train_transforms = [
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),
        ]
        self.test_transforms = []
        self.usual_mean = 0.
        self.usual_std = 0.
        self.non_sparse_mean = {}
        self.non_sparse_std = {}
        self.constant_pixel_val = 0.3

        # Further needed by the non-sparse mixin
        # These are transformed values of zero and one pixel values
        self.usual_zero = -0.4242129623889923
        self.usual_one = 2.821486711502075

        self.non_sparse_zero = {0.3: -1.7120479345321655}
        self.non_sparse_one = {0.3: 2.533843517303467}

    def get_dataset_(self, path, which, transform):
        return datasets.MNIST(path, train=(which == 'train'), transform=transform)

    def get_num_classes(self):
        return 10

    def get_each_entry_shape(self):
        return (1, 28, 28)

    def get_model(self, model_mode, device):
        if model_mode == 'fake-classes':
            model = ExampleCNNNet(20).to(device)
        elif model_mode == 'max-entropy':
            model = ExampleCNNNet(10).to(device)
        else:
            raise ValueError(f"Invalid model mode {model_mode}")
        # model = MLPNet().to(device)
        # model = MLPNet3Layer(num_classes=20).to(device)
        # model = MaxNormMLP(num_classes=20).to(device)
        return model

    def update_config(self, config):
        config.model_classname = 'ExampleCNNNet'

# class CIFARDatasetHelper(DatasetHelper):
#     def __init__(self, name, subset):
#         super().__init__(name, subset)
#
#     def get_transformed_zero_one(self):
#         raise NotImplementedError()
