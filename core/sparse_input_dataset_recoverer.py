import argparse
#import pytest
#print("module name: ", __name__)
#pytest.register_assert_rewrite(__name__)
#from unittest import TestCase

import torch

from core.sparse_input_recoverer import SparseInputRecoverer
from core.tblabels import TBLabels


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
        #self.dataset_epoch = 0
        self.image_zero = self.sparse_input_recoverer.image_zero
        self.image_one = self.sparse_input_recoverer.image_one

    def recover_image_dataset_internal(self, model, output_shape, num_real_classes, batch_size, num_steps,
                              include_layer_map, sparsity_mode, device, mode, dataset_epoch):
        assert output_shape[0] % batch_size == 0,\
            f"Number of images to generate not divisible by image recovery " \
            f"batch size: output_shape[0] = {output_shape[0]}, batch_size = {batch_size} "
        images = []
        targets = []
        batch_shape = list(output_shape)
        batch_shape[0] = batch_size
        num_batches = output_shape[0] // batch_size
        # Since we're separating plots for each batch, start can be kept 0
        start = 0  # self.dataset_epoch * num_batches
        end = start + num_batches
        # or 'all', which will include images, or 'none', which will not log anything
        self.sparse_input_recoverer.tensorboard_logging = 'stats_only' if mode == 'train' else 'none'
        for batch_idx in range(start, end):
            image_batch = torch.randn(batch_shape).to(device)
            targets_batch = torch.randint(low=0, high=num_real_classes, size=(batch_size,)).to(device)
            images.append(image_batch)
            targets.append(targets_batch)
            self.sparse_input_recoverer.tensorboard_label = \
                f"{TBLabels.RECOVERY_INTERNAL}/epoch_{dataset_epoch}/batch_{batch_idx}"
            self.sparse_input_recoverer.recover_image_batch(model, image_batch, targets_batch, num_steps,
                                                            include_layer_map[sparsity_mode],
                                                            sparsity_mode,
                                                            include_likelihood=True,
                                                            batch_idx=batch_idx)

        # Need toconcat the tensors and return
        with torch.no_grad():
            images_tensor = torch.cat(images)
            targets_tensor = torch.cat(targets)

            if mode == 'train': # Perform logging only for the train dataset
                def log_bin(suffix, bin):
                    self.log_regular_batch_stats(suffix, model, images_tensor[bin], targets_tensor[bin], include_layer_map,
                                                 sparsity_mode, dataset_epoch)

                #self.log_first_100_images_stats(model, images_tensor, targets_tensor, include_layer_map, sparsity_mode)
                self.log_regular_batch_stats('', model, images_tensor, targets_tensor, include_layer_map, sparsity_mode,
                                             dataset_epoch)
                # Bin images by probability and log
                bin_0_9 = (targets_tensor >= 0.9)
                bin_0_7_0_8 = (targets_tensor >= 0.7) & (targets_tensor < 0.9)
                bin_0_5_0_6 = (targets_tensor >= 0.5) & (targets_tensor < 0.7)
                bin_0_3_0_4 = (targets_tensor >= 0.3) & (targets_tensor < 0.5)
                bin_0_3 = (targets_tensor < 0.3)
                log_bin('prob_greater_than_eq_0.9', bin_0_9)
                log_bin('prob_between_0.7_0.9', bin_0_7_0_8)
                log_bin('prob_between_0.5_0.7', bin_0_5_0_6)
                log_bin('prob_between_0.3_0.5', bin_0_3_0_4)
                log_bin('prob_less_than_0.3', bin_0_3)

                # Log unconfident images

            # Save to ckpt dir
            self.ckpt_saver.save_images(mode, images_tensor, targets_tensor, dataset_epoch)

            #self.dataset_epoch += 1

        return images_tensor, targets_tensor

    def log_first_100_images_stats(self, model, images_tensor, targets_tensor, include_layer_map, sparsity_mode, dataset_epoch):
        # Add first 100 images to tensorboard
        n = images_tensor.shape[0]
        n = 100 if n > 100 else n
        first_100_images = torch.clone(images_tensor[0:n])
        first_100_targets = torch.clone(targets_tensor[0:n])
        first_100_targets_list = [foo.item() for foo in targets_tensor[0:n]]
        self.tbh.add_image_grid(first_100_images, f"{sparsity_mode}/dataset_images", filtered=True, num_per_row=10,
                                global_step=dataset_epoch)
        self.tbh.add_list(first_100_targets_list, f"{sparsity_mode}/dataset_targets", num_per_row=10,
                          global_step=dataset_epoch)
        # Run forward on this batch and get losses, probabilities and sparsity for logging
        loss, losses, output, probs, sparsity = self.sparse_input_recoverer.forward(
            model, first_100_images, first_100_targets, include_layer_map[sparsity_mode], include_likelihood=True)
        self.tbh.log_dict(f"{sparsity_mode}", probs, global_step=dataset_epoch)
        self.tbh.log_dict(f"{sparsity_mode}", sparsity, global_step=dataset_epoch)
        self.tbh.flush()

    def log_regular_batch_stats(self, suffix, model, images_tensor, targets_tensor, include_layer_map, sparsity_mode, dataset_epoch):
        prefix = TBLabels.RECOVERY_EPOCH #"recovery_epoch"
        img_label = f"{prefix}/dataset_images_{suffix}" if suffix else f"{prefix}/dataset_images"
        tgt_label = f"{prefix}/dataset_targets_{suffix}" if suffix else f"{prefix}/dataset_targets"

        images, targets = self.get_regular_batch(images_tensor, targets_tensor, self.num_real_classes, 10)
        targets_list = [foo.item() for foo in targets]
        self.tbh.add_image_grid(images, img_label, filtered=True, num_per_row=10,
                                global_step=dataset_epoch)
        self.tbh.add_list(targets_list, tgt_label, num_per_row=10,
                          global_step=dataset_epoch)
        # Run forward on this batch and get losses, probabilities and sparsity for logging
        # XXX: Skip this - we're going to do this on the full train data in
        #      AdversarialTrainer::generate_m_images_train_one_epoch_adversarial
        #
        # loss, losses, output, probs, sparsity = self.sparse_input_recoverer.forward(
        #     model, images, targets, include_layer_map[sparsity_mode], include_likelihood=True)
        # self.tbh.log_dict(f"{prefix}", probs, global_step=dataset_epoch)
        # self.tbh.log_dict(f"{prefix}", sparsity, global_step=dataset_epoch)
        self.tbh.flush()

    # Get a batch of 100 images with 10 images per class
    def get_regular_batch(self, images, targets, num_classes, num_per_class):
        entries = []
        tgt_entries = []
        for cls in range(num_classes):
            count = 0
            i = 0
            while count < num_per_class and i < targets.shape[0]:
                if targets[i].item() == cls:
                    entries.append(images[i])
                    tgt_entries.append(targets[i])
                    count += 1
                i += 1
            # Append cross X images if not enough entries for this class
            # All-zero images can be produced easily by our optimization algo,
            # But cross image is hard to be produced by accident
            for j in range(count, num_per_class):
                d1 = torch.diagflat(torch.ones(28))
                d2 = torch.flip(d1, dims=[1, ])
                cross = ((d1 + d2) > 0.).float().unsqueeze(dim=0)

                entries.append(cross * self.image_one + self.image_zero)
                tgt_entries.append(torch.tensor(cls, device=targets.device))

        return torch.stack(entries), torch.stack(tgt_entries)

    def recover_image_dataset(self, mode, dataset_epoch):
        output_shape = [self.dataset_len] + list(self.each_entry_shape)
        return self.recover_image_dataset_internal(self.model, output_shape, self.num_real_classes, self.batch_size,
                                                   self.num_recovery_steps, self.include_layer_map, self.sparsity_mode,
                                                   self.device, mode, dataset_epoch)

