from unittest import TestCase

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from sparse_recovery_main import setup_everything

import sys

class TestSparseInputDatasetRecoverer(TestCase):
    def test_recover_image_dataset(self):
        config, include_layer, labels, model, tbh, sparse_input_recoverer = setup_everything(sys.argv[2:])
        sparse_input_recoverer.tensorboard_logging = False
        dataset_recoverer = SparseInputDatasetRecoverer(sparse_input_recoverer)

        n = 20
        bs = 4
        images, targets = dataset_recoverer.recover_image_dataset(
            model, output_shape=(n, 1, 28, 28), num_classes=10, batch_size=bs, num_steps=10,
            include_layer=include_layer, sparsity_mode=config.penalty_mode)

        assert images.shape == (n, 1, 28, 28)
        assert targets.shape[0] == n

        print(targets.detach().numpy())
        #tbh.add_image_grid(images, "Final Dataset", filtered=False, global_step=None)
        #print("Added final images")
