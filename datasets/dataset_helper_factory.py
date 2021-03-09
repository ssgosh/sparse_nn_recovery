from datasets.cifar_dataset_helper import CIFARDatasetHelper
from datasets.mnist_dataset_helper import MNISTdatasetHelper
from datasets.mnist_tensor_dataset_helper import MNISTTensorDatasetHelper


class DatasetHelperFactory:
    dataset = None

    @classmethod
    def get(classobj, dataset_name : str = None):
        """
        Singleton method. Maintains application-wide train/test dataset

        :param dataset_name:
        :return:
        """
        if classobj.dataset is None:
            classobj.dataset = classobj.get_new(dataset_name)
            return classobj.dataset
        else:
            assert dataset_name is None
            return classobj.dataset

    @classmethod
    def get_new(classobj, cased_dataset_name):
        """
        Factory method to get new datasets.

        Various train/test datasets compatible to each other (such as MNIST, MNIST_A, MNIST_B etc) may be managed by
        creating new instances.

        :param cased_dataset_name:
        :return:
        """
        assert cased_dataset_name
        dataset_name = cased_dataset_name.lower()
        subset = dataset_name in ['mnist_a', 'mnist_b']
        non_sparse = 'non_sparse' in dataset_name
        if 'mnist' in dataset_name:
            return MNISTdatasetHelper(name=cased_dataset_name, subset=subset, non_sparse=non_sparse)
        elif 'cifar' in dataset_name:
            return CIFARDatasetHelper(name=cased_dataset_name, subset=subset, non_sparse=non_sparse)
        elif dataset_name == 'external_b':
            return MNISTTensorDatasetHelper(name=cased_dataset_name)
        else:
            raise ValueError("Invalid dataset name: %s" % dataset_name)
