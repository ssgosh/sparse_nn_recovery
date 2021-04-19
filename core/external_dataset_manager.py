from torch.utils.data import DataLoader

from datasets.dataset_helper_factory import DatasetHelperFactory


class ExternalDatasetManager:
    def __init__(self, test_batch_size, config):
        self.test_batch_size = test_batch_size

        self.test_loaders = []
        self.test_names = []

        self.valid_loaders = []
        self.valid_names = []

        if config.dataset.lower() not in ['cifar']:
            if not config.non_sparse_dataset:
                self.add_dataset('external_B')
            else:
                self.add_dataset('external_B_non_sparse')

    def add_dataset(self, dataset_name):
        self.add_dataset_(dataset_name, which='test')
        self.add_dataset_(dataset_name, which='valid')

    def add_dataset_(self, dataset_name, which):
        if which == 'test':
            loaders = self.test_loaders
            names = self.test_names
        elif which == 'valid':
            loaders = self.valid_loaders
            names = self.valid_names
        else:
            assert False, f"Invalid value for param which : {which}"

        loaders.append(
            DataLoader(
                DatasetHelperFactory.get_new(dataset_name, non_sparse=False).get_dataset(which),
                **{'batch_size' : self.test_batch_size}
            )
        )
        names.append(dataset_name)
