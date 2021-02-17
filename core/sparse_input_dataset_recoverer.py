import torch

from core.sparse_input_recoverer import SparseInputRecoverer


class SparseInputDatasetRecoverer:

    def __init__(self, config, tbh, verbose=False):
        self.config = config
        self.tbh = tbh
        self.verbose = verbose
        self.sparse_input_recoverer = SparseInputRecoverer(config, tbh, verbose)

    def recover_image_dataset(self, model, output_shape, num_classes, batch_size, num_steps, include_layer, sparsity_mode):
        assert output_shape[0] % batch_size == 0
        images = []
        targets = []
        batch_shape = list(output_shape)
        batch_shape[0] = batch_size
        for batch_idx in range(output_shape[0] / batch_size):
            image_batch = torch.randn(batch_shape)
            targets_batch = torch.randint(low=0, high=num_classes, size=batch_size)
            images.append(image_batch)
            targets.append(targets_batch)
            self.sparse_input_recoverer.recover_image_batch(self, model, image_batch, targets_batch, num_steps, include_layer,
                                                            sparsity_mode,
                                                            include_likelihood=True)

        # Need to concat the tensors and return
        with torch.no_grad():
            images_tensor = torch.cat(images)
            targets_tensor = torch.cat(targets)

        return images_tensor, targets_tensor
