from datasets.dataset_helper_factory import DatasetHelperFactory

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

import unittest


class MyTestCase(unittest.TestCase):
    def test_get_batched_epsilon(self):
        dh = DatasetHelperFactory.get('cifar')
        dh.get_batched_epsilon()
        dh = DatasetHelperFactory.get_new('mnist', non_sparse=False)
        dh.get_batched_epsilon()



if __name__ == '__main__':
    unittest.main()
