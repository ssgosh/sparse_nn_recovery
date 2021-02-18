from unittest import TestCase

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from sparse_recovery_main import setup_everything

import sys

class TestSparseInputDatasetRecoverer(TestCase):
    def test_recover_image_dataset(self):
        config, include_layer, labels, model, tbh, sparse_input_recoverer = setup_everything(sys.argv[2:])
        dataset_recoverer = SparseInputDatasetRecoverer(sparse_input_recoverer)

        images, targets = dataset_recoverer.recover_image_dataset(
            model, output_shape=(10, 1, 28, 28), num_classes=10, batch_size=5, num_steps=100,
            include_layer=include_layer, sparsity_mode=config.penalty_mode)

        assert images.shape == (10, 1, 28, 28)
        assert targets.shape[0] == 10

        tbh.add_image_grid(images, "Final Dataset", filtered=False, global_step=None)
