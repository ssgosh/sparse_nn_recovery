from datasets.dataset_helper import CIFARDatasetHelper
from datasets.mnist_dataset_helper import MNISTdatasetHelper


class DatasetHelperFactory:
    dataset = None

    @classmethod
    def get(classobj, dataset_name : str = None):
        """
        Singleton method

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
        Factory method to get new datasets

        :param dataset_name:
        :return:
        """
        assert cased_dataset_name
        dataset_name = cased_dataset_name.lower()
        subset = dataset_name not in ['mnist', 'cifar']
        if 'mnist' in dataset_name:
            return MNISTdatasetHelper(name=cased_dataset_name, subset=subset)
        elif 'cifar' in dataset_name:
            return CIFARDatasetHelper(name=cased_dataset_name, subset=subset)
        else:
            raise ValueError("Invalid dataset name: %s" % dataset_name)