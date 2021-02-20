import torch

from core.sparse_input_recoverer import SparseInputRecoverer


# Creates an entire dataset of images, targets by doing sparse recovery from the model.
# Note that targets has the actual class for which the images were trained. In order to
# Use this later in adversarial training, please add num_real_classes to targets to create fake
# class targets
class SparseInputDatasetRecoverer:

    def __init__(self, sparse_input_recoverer : SparseInputRecoverer, model, num_recovery_steps, batch_size,
                 sparsity_mode, num_real_classes, dataset_len, each_entry_shape):
        self.sparse_input_recoverer = sparse_input_recoverer
        self.model = model
        self.include_layer_map = SparseInputRecoverer.include_layer
        self.sparsity_mode = sparsity_mode
        self.num_recovery_steps = num_recovery_steps
        self.batch_size = batch_size
        self.num_real_classes = num_real_classes
        self.dataset_len = dataset_len
        self.each_entry_shape = each_entry_shape

    def recover_image_dataset_internal(self, model, output_shape, num_real_classes, batch_size, num_steps,
                              include_layer, sparsity_mode):
        assert output_shape[0] % batch_size == 0
        images = []
        targets = []
        batch_shape = list(output_shape)
        batch_shape[0] = batch_size
        for batch_idx in range(output_shape[0] // batch_size):
            image_batch = torch.randn(batch_shape)
            targets_batch = torch.randint(low=0, high=num_real_classes, size=(batch_size,))
            images.append(image_batch)
            targets.append(targets_batch)
            self.sparse_input_recoverer.recover_image_batch(model, image_batch, targets_batch, num_steps,
                                                            include_layer[sparsity_mode],
                                                            sparsity_mode,
                                                            include_likelihood=True,
                                                            batch_idx=batch_idx)

        # Need to concat the tensors and return
        with torch.no_grad():
            images_tensor = torch.cat(images)
            targets_tensor = torch.cat(targets)

        return images_tensor, targets_tensor

    def recover_image_dataset(self):
        output_shape = [self.dataset_len] + list(self.each_entry_shape)
        return self.recover_image_dataset_internal(self.model, output_shape, self.num_real_classes, self.batch_size,
                                                   self.num_recovery_steps, self.include_layer_map, self.sparsity_mode)
