from datasets.dataset_helper import DatasetHelper


class CIFARDatasetHelper(DatasetHelper):
    def __init__(self, name, subset):
        super().__init__(name, subset)

    def get_transformed_zero_one(self):
        raise NotImplementedError()