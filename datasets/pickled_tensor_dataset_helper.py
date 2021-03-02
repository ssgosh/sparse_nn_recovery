import pathlib

import torch
from torch.utils.data import TensorDataset

from datasets.mnist_dataset_helper import MNISTdatasetHelper


class PickledTensorDatasetHelperMixin:
    """
    This is a Mixin class which provides the get_tensor_dataset function.
    This will be used to create MNISTTensorDatasetHelper, CIFARTensorDatasetHelper etc.
    Because of duck typing (quack, quack!!!), we can assume members which are not defined here.
    We also don't need to subclass anything. Subclassing DatasetHelper would be problematic since then we'll have the
    diamond problem when we try to subclass both MNISTDatasetHelper and this class to create MNISTTensorDatasetHelper.
    """

    def get_tensor_dataset(self, path, which):
        """
        Expected members: self.name,
        :return: A tensor dataset loaded from disk
        """
        fname = pathlib.Path(path) / self.name / f"{which}.pt"
        d = torch.load(fname)
        images = d['images']
        targets = d['targets']
        return TensorDataset(images, targets)
