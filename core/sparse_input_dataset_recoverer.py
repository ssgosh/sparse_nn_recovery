import argparse
#import pytest
#print("module name: ", __name__)
#pytest.register_assert_rewrite(__name__)
#from unittest import TestCase

import torch
from icontract import ensure

from core.sparse_input_recoverer import SparseInputRecoverer
from core.tblabels import TBLabels
from datasets.dataset_helper_factory import DatasetHelperFactory
from utils import image_processor

from utils.torchutils import get_cross, safe_clone


# Creates an entire dataset of images, targets by doing sparse recovery from the model.
# Note that targets has the actual class for which the images were trained. In order to
# Use this later in adversarial training, please add num_real_classes to targets to create fake
# class targets
class SparseInputDatasetRecoverer:

    @staticmethod
    def add_command_line_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--recovery-batch-size', type=int, default=1024, required=False, metavar='N',
                            help='Batch size for image generation')
        parser.add_argument('--recovery-prune', action='store_true', dest='recovery_prune', default=True,
                            required=False, help='Prune Low Probability or non-sparse Images from adversarial dataset')
        parser.add_argument('--no-recovery-prune', action='store_false', dest='recovery_prune', default=True,
                            required=False, help='Disable pruning of low probability or non-sparse images from adversarial dataset')
        parser.add_argument('--recovery-low-prob-threshold', type=float, default=0.9, required=False,
                            help='Generated adversarial images with probability less than this will be pruned')
        parser.add_argument('--recovery-sparsity-threshold', type=int, default=100, required=False,
                            help='Generated adversarial images with sparsity greater than this will be pruned')

    def __init__(self, sparse_input_recoverer : SparseInputRecoverer, model, num_recovery_steps, batch_size,
                 sparsity_mode, num_real_classes, dataset_len, each_entry_shape, device, ckpt_saver, config):
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
        self.batched_image_zero = self.sparse_input_recoverer.batched_image_zero
        self.batched_image_one = self.sparse_input_recoverer.batched_image_one

        # Prune the recovered dataset for low-probability images
        self.prune = config.recovery_prune
        self.low_prob_threshold = config.recovery_low_prob_threshold
        self.sparsity_threshold = config.recovery_sparsity_threshold

    def recover_image_dataset_internal(self, model, output_shape, num_real_classes, batch_size, num_steps,
                                       include_layer_map, sparsity_mode, device, mode, dataset_epoch, prune):
        assert output_shape[0] % batch_size == 0,\
            f"Number of images to generate not divisible by image recovery " \
            f"batch size: output_shape[0] = {output_shape[0]}, batch_size = {batch_size} "
        images = []
        targets = []
        probs = []
        batch_shape = list(output_shape)
        batch_shape[0] = batch_size
        num_batches = output_shape[0] // batch_size
        # Since we're separating plots for each batch, start can be kept 0
        start = 0  # self.dataset_epoch * num_batches
        end = start + num_batches
        # or 'all', which will include images, or 'none', which will not log anything
        self.sparse_input_recoverer.tensorboard_logging = 'stats_only' if mode == 'train' else 'none'
        # Disable this because tensorboard files are growing too large
        # and we don't seem to be focusing on these internal stats anyway
        # self.sparse_input_recoverer.tensorboard_logging = 'none'
        for batch_idx in range(start, end):
            image_batch = torch.randn(batch_shape).to(device)
            targets_batch = torch.randint(low=0, high=num_real_classes, size=(batch_size,)).to(device)
            images.append(image_batch)
            targets.append(targets_batch)
            self.sparse_input_recoverer.tensorboard_label = \
                f"{TBLabels.RECOVERY_INTERNAL}/epoch_{dataset_epoch}/batch_{batch_idx}"
            probs_batch = self.sparse_input_recoverer.recover_image_batch(model, image_batch, targets_batch, num_steps,
                                                            include_layer_map[sparsity_mode],
                                                            sparsity_mode,
                                                            include_likelihood=True,
                                                            batch_idx=batch_idx)
            #print("probs_batch :", probs_batch)
            probs.append(probs_batch)

        # Need toconcat the tensors and return
        with torch.no_grad():
            images_tensor = torch.cat(images)
            targets_tensor = torch.cat(targets)
            probs_tensor = torch.cat(probs)

            if mode == 'train': # Perform logging only for the train dataset
                def log_bin(suffix, bin):
                    self.tbh.log_regular_batch_stats('adv', suffix, model, images_tensor[bin], targets_tensor[bin], include_layer_map,
                                                     sparsity_mode, dataset_epoch)

                #self.log_first_100_images_stats(model, images_tensor, targets_tensor, include_layer_map, sparsity_mode)
                self.tbh.log_regular_batch_stats('adv', '', model, images_tensor, targets_tensor, include_layer_map, sparsity_mode,
                                                 dataset_epoch)
                # Bin images by probability and log
                bin_0_9 = (probs_tensor >= 0.9)
                bin_0_7_0_8 = (probs_tensor >= 0.7) & (probs_tensor < 0.9)
                bin_0_5_0_6 = (probs_tensor >= 0.5) & (probs_tensor < 0.7)
                bin_0_3_0_4 = (probs_tensor >= 0.3) & (probs_tensor < 0.5)
                bin_0_3 = (probs_tensor < 0.3)
                log_bin('prob_greater_than_eq_0.9', bin_0_9)
                log_bin('prob_between_0.7_0.9', bin_0_7_0_8)
                log_bin('prob_between_0.5_0.7', bin_0_5_0_6)
                log_bin('prob_between_0.3_0.5', bin_0_3_0_4)
                log_bin('prob_less_than_0.3', bin_0_3)

                # Grab 2 images per class per bin
                def log_images_sorted(img_per_bin=2):
                    num_per_class1 = 5 * img_per_bin # 5 bins

                    @ensure(lambda result : len(result) == img_per_bin)
                    def get_cross_if_empty(imgs1, bin1, tgts1, cls1):
                        x = imgs1[bin1 & (tgts1 == cls1)]
                        ret = []
                        i = 0
                        for img in x:
                            ret.append(img)
                            i += 1
                            if i >= img_per_bin:
                                break
                        while i < img_per_bin:
                            shape = DatasetHelperFactory.get().get_each_entry_shape()
                            ret.append(self.batched_image_one.squeeze(dim=0) * get_cross(shape[2], shape[0], imgs1) + self.batched_image_zero.squeeze(dim=0))
                            i += 1
                        return ret

                    per_class_bin = {
                        "bin_0_9" : [],
                        "bin_0_7_0_8" : [],
                        "bin_0_5_0_6" : [],
                        "bin_0_3_0_4" : [],
                        "bin_0_3" : [],
                    }
                    images1 = []
                    for cls in range(self.num_real_classes):
                        # per_class_bin["bin_0_9"].append(   images_tensor[   bin_0_9 & (targets_tensor == cls)   ] )
                        # per_class_bin["bin_0_7_0_8"].append(   images_tensor[   bin_0_7_0_8 & (targets_tensor == cls)   ] )
                        # per_class_bin["bin_0_5_0_6"].append(   images_tensor[   bin_0_5_0_6 & (targets_tensor == cls)   ] )
                        # per_class_bin["bin_0_3_0_4"].append(   images_tensor[   bin_0_3_0_4 & (targets_tensor == cls)   ] )
                        # per_class_bin["bin_0_3"].append(   images_tensor[   bin_0_3 & (targets_tensor == cls)   ] )

                        images1.extend(get_cross_if_empty(images_tensor, bin_0_3, targets_tensor, cls))
                        images1.extend(get_cross_if_empty(images_tensor, bin_0_3_0_4, targets_tensor, cls))
                        images1.extend(get_cross_if_empty(images_tensor, bin_0_5_0_6, targets_tensor, cls))
                        images1.extend(get_cross_if_empty(images_tensor, bin_0_7_0_8, targets_tensor, cls))
                        images1.extend(get_cross_if_empty(images_tensor, bin_0_9, targets_tensor, cls))

                    images1_tensor = torch.stack(images1, dim=0)
                    targets1_tensor = torch.tensor([cls for cls in range(self.num_real_classes)],
                                                   device=targets_tensor.device)
                    self.tbh.log_regular_batch_stats('adv', 'sorted', model, images1_tensor, targets1_tensor, include_layer_map,
                                                 sparsity_mode, dataset_epoch, precomputed=True)
                # Log unconfident images
                log_images_sorted()

            sparsity_tensor = image_processor.get_sparsity_batch(images_tensor, self.batched_image_zero)
            # Save to ckpt dir
            self.ckpt_saver.save_images(mode, '', images_tensor, targets_tensor, probs_tensor, sparsity_tensor, dataset_epoch)

            if prune:
                images_tensor, targets_tensor, probs_tensor, sparsity_tensor = self.prune_images(images_tensor, targets_tensor, probs_tensor, sparsity_tensor)
                self.ckpt_saver.save_images(mode, 'pruned', images_tensor, targets_tensor, probs_tensor, sparsity_tensor, dataset_epoch)
            #self.dataset_epoch += 1

        return images_tensor, targets_tensor, probs_tensor

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


    def recover_image_dataset(self, mode, dataset_epoch):
        output_shape = [self.dataset_len] + list(self.each_entry_shape)
        return self.recover_image_dataset_internal(self.model, output_shape, self.num_real_classes, self.batch_size,
                                                   self.num_recovery_steps, self.include_layer_map, self.sparsity_mode,
                                                   self.device, mode, dataset_epoch, self.prune)

    def prune_images(self, images_tensor, targets_tensor, probs_tensor, sparsity_tensor):
        keep = (probs_tensor >= self.low_prob_threshold)
        keep1 = (sparsity_tensor <= self.sparsity_threshold)
        keep = keep * keep1
        img = safe_clone(images_tensor[keep])
        tgt = safe_clone(targets_tensor[keep])
        probs = safe_clone(probs_tensor[keep])
        sparsity = safe_clone(sparsity_tensor[keep])
        return img, tgt, probs, sparsity

