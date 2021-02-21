import unittest

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from unittest import TestCase

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from sparse_recovery_main import setup_everything

import sys

class TestSparseInputDatasetRecoverer(TestCase):
    def test_recover_image_dataset(self):
        config, include_layer, labels, model, tbh, sparse_input_recoverer = setup_everything(sys.argv[2:])
        sparse_input_recoverer.tensorboard_logging = False
        n = 20
        bs = 4
        dataset_recoverer = SparseInputDatasetRecoverer(sparse_input_recoverer, model, num_recovery_steps=10,
                                                        batch_size=bs, sparsity_mode=config.recovery_penalty_mode,
                                                        num_real_classes=10, dataset_len=n,
                                                        each_entry_shape=(1, 28, 28), device='cpu')

        images, targets = dataset_recoverer.recover_image_dataset()

        assert images.shape == (n, 1, 28, 28)
        assert targets.shape[0] == n
        print(targets.detach().numpy())

        images, targets = dataset_recoverer.recover_image_dataset_internal(model, output_shape=(100, 1, 28, 28),
                                                                           num_real_classes=5, batch_size=10,
                                                                           num_steps=10, include_layer=include_layer,
                                                                           sparsity_mode=config.recovery_penalty_mode,
                                                                           device='cpu')
        assert images.shape == (100, 1, 28, 28)
        assert targets.shape[0] == 100

        print(targets.detach().numpy())
        #tbh.add_image_grid(images, "Final Dataset", filtered=False, global_step=None)
        #print("Added final images")

if __name__ == "__main__":
    unittest.main()