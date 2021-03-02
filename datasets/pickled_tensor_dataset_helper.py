import pathlib

from datasets.mnist_dataset_helper import MNISTdatasetHelper


class PickledTensorDatasetHelperMixin:

    def get_tensor_dataset(self, path, which):
        """
        Expected members: self.name,
        :return: A tensor dataset loaded from disk
        """
        fname = pathlib.Path(path) / self.name
        d = torch.load(fname)
        images = d['images']
        targets = d['targets']
