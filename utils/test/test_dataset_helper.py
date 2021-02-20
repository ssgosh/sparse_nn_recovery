import unittest

if __name__ == '__main__':
    import sys
    sys.path.append(".")

from utils.dataset_helper import *

class DatasetHelperTest(unittest.TestCase):
    def test_dataset_helper(self):
        self.assertRaises(AssertionError, DatasetHelper.get_dataset )
        dataset_helper = DatasetHelper.get_dataset('mnist')
        self.assertIsNotNone(dataset_helper)
        self.assertEqual(type(dataset_helper), MNISTdatasetHelper)
        zero, one = dataset_helper.get_transformed_zero_one()
        mnist_zero, mnist_one = mnist_helper.compute_mnist_transform_low_high()
        self.assertEqual(zero, mnist_zero)
        self.assertEqual(one, mnist_one)
        self.assertRaises(AssertionError, DatasetHelper.get_dataset)


if __name__ == '__main__':
    unittest.main()
