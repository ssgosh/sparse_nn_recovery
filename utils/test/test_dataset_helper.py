import unittest

if __name__ == '__main__':
    import sys
    sys.path.append(".")

from datasets.dataset_helper import *

class DatasetHelperTest(unittest.TestCase):
    def test_dataset_helper_mnist(self):
        self.assertRaises(AssertionError, DatasetHelper.get)
        dataset_helper = DatasetHelper.get('mnist')
        self.assertIsNotNone(dataset_helper)
        self.assertEqual(type(dataset_helper), MNISTdatasetHelper)
        zero, one = dataset_helper.get_transformed_zero_one()
        mnist_zero, mnist_one = mnist_helper.compute_mnist_transform_low_high()
        self.assertEqual(zero, mnist_zero)
        self.assertEqual(one, mnist_one)
        self.assertRaises(AssertionError, DatasetHelper.get)

    def test_dataset_helper_cifar(self):
        dataset_helper = DatasetHelper.get('cifar')
        self.assertIsNotNone(dataset_helper)
        self.assertEqual(type(dataset_helper), CIFARDatasetHelper)
        dataset_helper.get_transformed_zero_one()




if __name__ == '__main__':
    unittest.main()
