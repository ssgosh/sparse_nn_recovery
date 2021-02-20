from abc import ABC, abstractmethod

# Abstraction for Dataset-specific functionality, such as transformations,
# values of transformed zero and one pixel values.
# Used in the main scripts
class DatasetHelper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_transformed_zero_one(self):
        pass


class