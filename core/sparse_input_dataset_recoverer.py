import argparse
#import pytest
#print("module name: ", __name__)
#pytest.register_assert_rewrite(__name__)
#from unittest import TestCase

import torch

from core.sparse_input_recoverer import SparseInputRecoverer


# Creates an entire dataset of images, targets by doing sparse recovery from the model.
# Note that targets has the actual class for which the images were trained. In order to
# Use this later in adversarial training, please add num_real_classes to targets to create fake
# class targets
class SparseInputDatasetRecoverer:

    @staticmethod
    def add_command_line_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--recovery-batch-size', type=int, default=1024, required=False, metavar='N',
                            help='Batch size for image generation')

    def __init__(self, sparse_input_recoverer : SparseInputRecoverer, model, num_recovery_steps, batch_size,
                 sparsity_mode, num_real_classes, dataset_len, each_entry_shape, device, ckpt_saver):
        self.sparse_input_recoverer = sparse_input_recoverer
        self.model = model
        self.include_layer_map = SparseInputRecoverer.include_layer_map
        self.sparsity_mode = sparsity_mode
        self.num_recovery_steps = num_recovery_steps
        self.batch_size = batch_size
        self.num_real_classes = num_real_classes
        self.dataset_len = dataset_len
        self.each_entry_shape = each_entry_shape
        self.device = device
        self.ckpt_saver = ckpt_saver
        self.tbh = self.sparse_input_recoverer.tbh
        self.dataset_epoch = 0

    def recover_image_dataset_internal(self, model, output_shape, num_real_classes, batch_size, num_steps,
                              include_layer_map, sparsity_mode, device):
        assert output_shape[0] % batch_size == 0,\
            f"Number of images to generate not divisible by image recovery " \
            f"batch size: output_shape[0] = {output_shape[0]}, batch_size = {batch_size} "
        images = []
        targets = []
        batch_shape = list(output_shape)
        batch_shape[0] = batch_size
        self.sparse_input_recoverer.tensorboard_logging = False
        for batch_idx in range(output_shape[0] // batch_size):
            image_batch = torch.randn(batch_shape).to(device)
            targets_batch = torch.randint(low=0, high=num_real_classes, size=(batch_size,)).to(device)
            images.append(image_batch)
            targets.append(targets_batch)
            self.sparse_input_recoverer.recover_image_batch(model, image_batch, targets_batch, num_steps,
                                                            include_layer_map[sparsity_mode],
                                                            sparsity_mode,
                                                            include_likelihood=True,
                                                            batch_idx=batch_idx)

        # Need to concat the tensors and return
        with torch.no_grad():
            images_tensor = torch.cat(images)
            targets_tensor = torch.cat(targets)

            self.log_first_100_images_stats(model, images_tensor, targets_tensor, include_layer_map, sparsity_mode)

            # Save to ckpt dir
            self.ckpt_saver.save_images(images_tensor, targets_tensor, self.dataset_epoch)

            self.dataset_epoch += 1

        return images_tensor, targets_tensor

    def log_first_100_images_stats(self, model, images_tensor, targets_tensor, include_layer_map, sparsity_mode):
        # Add first 100 images to tensorboard
        n = images_tensor.shape[0]
        n = 100 if n > 100 else n
        first_100_images = torch.clone(images_tensor[0:n])
        first_100_targets = torch.clone(targets_tensor[0:n])
        first_100_targets_list = [foo.item() for foo in targets_tensor[0:n]]
        self.tbh.add_image_grid(first_100_images, f"{sparsity_mode}/dataset_images", filtered=True, num_per_row=10,
                                global_step=self.dataset_epoch)
        self.tbh.add_list(first_100_targets_list, f"{sparsity_mode}/dataset_targets", num_per_row=10,
                          global_step=self.dataset_epoch)
        # Run forward on this batch and get losses, probabilities and sparsity for logging
        loss, losses, output, probs, sparsity = self.sparse_input_recoverer.forward(
            model, first_100_images, first_100_targets, include_layer_map[sparsity_mode], include_likelihood=True)
        self.tbh.log_dict(f"{sparsity_mode}", probs, global_step=self.dataset_epoch)
        self.tbh.log_dict(f"{sparsity_mode}", sparsity, global_step=self.dataset_epoch)
        self.tbh.flush()

    def recover_image_dataset(self):
        output_shape = [self.dataset_len] + list(self.each_entry_shape)
        return self.recover_image_dataset_internal(self.model, output_shape, self.num_real_classes, self.batch_size,
                                                   self.num_recovery_steps, self.include_layer_map, self.sparsity_mode,
                                                   self.device)
