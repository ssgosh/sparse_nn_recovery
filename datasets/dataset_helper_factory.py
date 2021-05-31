import argparse

from datasets.cifar_imagenet_dataset_helper import CIFARImageNetDatasetHelper
from datasets.cifar_dataset_helper import CIFARDatasetHelper
from datasets.dataset_helper import DatasetHelper
from datasets.mnist_dataset_helper import MNISTdatasetHelper
from datasets.mnist_tensor_dataset_helper import MNISTTensorDatasetHelper


class DatasetHelperFactory:
    dataset = None

    @staticmethod
    def add_command_line_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--non-sparse-dataset', action='store_true', default=True, dest='non_sparse_dataset',
                            help='Load dataset in non-sparse mode')
        parser.add_argument('--sparse-dataset', action='store_false', default=True, dest='non_sparse_dataset',
                            help='Load dataset in sparse mode')
        parser.add_argument('--use-imagenet-pretrained-model', action='store_true', default=False,
                dest='use_imagenet_pretrained_model',
                help='Use model pretrained on ImageNet')

    @classmethod
    def get(classobj, dataset_name : str = None, non_sparse : bool = False) -> 'DatasetHelper':
        """
        Singleton method. Maintains application-wide train/test dataset

        :param dataset_name:
        :param non_sparse:
        :return:
        """
        if classobj.dataset is None:
            classobj.dataset = classobj.get_new(dataset_name, non_sparse)
            return classobj.dataset
        else:
            assert dataset_name is None
            return classobj.dataset

    @classmethod
    def get_new(classobj, cased_dataset_name, non_sparse):
        """
        Factory method to get new datasets.

        Various train/test datasets compatible to each other (such as MNIST, MNIST_A, MNIST_B etc) may be managed by
        creating new instances.

        :param cased_dataset_name:
        :param non_sparse:
        :return:
        """
        assert cased_dataset_name
        dataset_name = cased_dataset_name.lower()
        subset = dataset_name in ['mnist_a', 'mnist_b']
        # non_sparse = 'non_sparse' in dataset_name
        if 'mnist' in dataset_name:
            return MNISTdatasetHelper(name=cased_dataset_name, subset=subset, non_sparse=non_sparse)
        elif 'cifar_imagenet' in dataset_name:
            return CIFARImageNetDatasetHelper(name=cased_dataset_name, subset=subset, non_sparse=non_sparse)
        elif 'cifar' in dataset_name:
            return CIFARDatasetHelper(name=cased_dataset_name, subset=subset, non_sparse=non_sparse)
        elif dataset_name == 'external_b':
            # This dataset is sparse
            # This is needed so that appropriate mean and std values may be set
            assert non_sparse is False
            return MNISTTensorDatasetHelper(name=cased_dataset_name, non_sparse=non_sparse)
        elif dataset_name == 'external_b_non_sparse':
            # This dataset is non-sparse. Make sure the intent is clear
            # This is needed so that appropriate mean and std values may be set for this dataset
            assert non_sparse is True
            return MNISTTensorDatasetHelper(name=cased_dataset_name, non_sparse=non_sparse)
        else:
            raise ValueError("Invalid dataset name: %s" % dataset_name)
