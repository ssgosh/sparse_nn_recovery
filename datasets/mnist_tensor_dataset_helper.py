from datasets.mnist_dataset_helper import MNISTdatasetHelper
from datasets.pickled_tensor_dataset_helper import PickledTensorDatasetHelperMixin


class MNISTTensorDatasetHelper(MNISTdatasetHelper, PickledTensorDatasetHelperMixin):
    def __init__(self, name):
        super().__init__(name, subset=False)

    def get_dataset_(self, path, which, transform):
        """
        Override the implementation in MNISTdatasetHelper and call PickledTensorDatasetHelperMixin.get_tensor_dataset()
        instead.
        """
        assert transform is None
        return self.get_tensor_dataset(path, which)
