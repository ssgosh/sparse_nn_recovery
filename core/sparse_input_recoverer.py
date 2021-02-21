from __future__ import print_function

import argparse
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import image_processor as imp
from utils import plotter
from utils.metrics_helper import MetricsHelper
from utils.model_context_manager import model_eval_no_grad, images_require_grad


class SparseInputRecoverer:

    # Dictionary of string penalty modes to array of boolean values,
    # indicating whether penalty is on/off for the corresponding layer in that mode
    include_layer = {
        "no penalty": [False, False, False, False],
        "input only": [True, False, False, False],
        "all layers": [True, True, True, True],
        "layer 1 only": [False, True, False, False],
        "layer 2 only": [False, False, True, False],
        "layer 3 only": [False, False, False, True],
        "all but input": [False, True, True, True],
    }

    # All available penalty modes
    all_penalty_modes = list(include_layer.keys())

    @staticmethod
    def add_sparse_recovery_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--num-recovery-steps', type=int, default=1000, required=False,
                            help='Number of steps of gradient descent for image generation')
        parser.add_argument('--lambd', type=float, metavar='L',
                            default=0.1, required=False,
                            help='L1 penalty lambda on each layer')
        parser.add_argument('--penalty-mode', type=str, default='input only', required=False,
                            help='When mode is single-digit, which penalty mode should be used')
        parser.add_argument('--disable-pgd', dest='use_pgd', action='store_false',
                            default=True, required=False,
                            help='Disable Projected Gradient Descent (clipping)')
        parser.add_argument('--enable-pgd', dest='use_pgd', action='store_true',
                            default=True, required=False,
                            help='Enable Projected Gradient Descent (clipping)')


    @staticmethod
    def setup_default_config(config):
        config.include_layer = SparseInputRecoverer.include_layer
        config.labels = SparseInputRecoverer.all_penalty_modes
        config.include_likelihood = True
        config.lambd_layers = 3 * [config.lambd]  # [0.1, 0.1, 0.1]

    def __init__(self, config, tbh, verbose=False):
        """

        :type tbh: TensorBoardHelper
        :type verbose: bool
        """
        self.verbose = verbose
        self.config = config
        self.tbh = tbh
        self.image_zero = config.image_zero
        self.image_one = config.image_one
        self.metrics_helper = MetricsHelper(self.image_zero, self.image_one)
        self.tensorboard_logging = True

    # Clip the pixels to between (mnist_zero, mnist_one)
    def clip_if_needed(self, images):
        if self.config.use_pgd:
            with torch.no_grad():
                torch.clip(images, self.image_zero, self.image_one, out=images)

    # include_layer: boolean vector of whether to include a layer's l1 penalty
    def recover_image_batch(self, model, images, targets, num_steps, include_layer, penalty_mode,
                            include_likelihood=True, batch_idx=0):
        with model_eval_no_grad(model), images_require_grad(images):
            self.recover_image_batch_internal(model, images, targets, num_steps, include_layer, penalty_mode,
                                             include_likelihood, batch_idx)

    def recover_image_batch_internal(self, model, images, targets, num_steps, include_layer, penalty_mode,
                            include_likelihood, batch_idx):
        optimizer = optim.Adam([images], lr=0.5)

        # lambda for input
        lambd = self.config.lambd
        # lambd = 0.01
        # lambda for each layer
        lambd_layers = self.config.lambd_layers  # [0.1, 0.1, 0.1]
        # lambd_layers = [0.01, 0.01, 0.01]
        # lambd2 = 1.
        start = num_steps * batch_idx + 1
        for i in range(start, start + num_steps):
            losses = {}
            probs = {}
            sparsity = {}
            optimizer.zero_grad()
            output = model(images)
            if include_likelihood:
                nll_loss = F.nll_loss(output, targets)
            else:
                nll_loss = torch.tensor(0.)

            losses[f"nll_loss"] = nll_loss.item()

            # include l1 penalty only if it's given as true for that layer
            l1_loss = torch.tensor(0.)
            if include_layer[0]:
                l1_loss = lambd * (torch.norm(images - self.image_zero, 1)
                                   / torch.numel(images))

            losses[f"input_l1_loss"] = l1_loss.item()

            l1_layers = torch.tensor(0.)
            for idx, (include, lamb, l1) in enumerate(zip(include_layer[1:], lambd_layers,
                                                          model.all_l1)):
                if include:
                    layer_loss = lamb * l1
                    l1_layers += layer_loss
                    losses[f"layer_{idx}_l1_loss"] = layer_loss.item()

            losses[f"hidden_layers_l1_loss"] = l1_layers.item()
            losses[f"all_layers_l1_loss"] = l1_loss.item() + l1_layers.item()

            loss = nll_loss + l1_loss + l1_layers
            loss.backward()

            # Do step before computation of metrics which will change on backprop
            optimizer.step()
            self.clip_if_needed(images)

            losses[f"total_loss"] = loss.item()

            self.metrics_helper.compute_probs(output, probs, targets)
            self.metrics_helper.compute_sparsities(images, model, targets, sparsity)

            if self.verbose:
                print("Iter: ", i, ", Loss: %.3f" % loss.item(),
                      f"Prob of {targets[0]} %.3f" %
                      pow(math.e, output[0][targets[0].item()].item()),
                      "images median, mean, std, min, max: %.3f, %.3f, %.3f, %.3f, %.3f" % (
                          images.median().item(), images.mean().item(), images.std().item(), images.min().item(),
                          images.max().item()))

            if self.tensorboard_logging:
                # Do tensorboard things
                self.tbh.add_tensorboard_stuff(penalty_mode, model, images, losses, probs,
                                               sparsity, i)


    # Single digit, single label
    def recover_and_plot_single_digit(self, initial_image, label, targets, include_layer, model):
        self.recover_image_batch(model, initial_image, targets, self.config.num_recovery_steps, include_layer[label], label,
                                 include_likelihood=True)
        plotter.plot_single_digit(initial_image.detach()[0][0], targets[0], label,
                                  filtered=False)
        plotter.plot_single_digit(initial_image.detach()[0][0], targets[0], label,
                                  filtered=True)
        return initial_image

    # The main loop
    #
    # Uses recover_image() to perform sparse recovery of digit images from a model
    def recover_and_plot_images_varying_penalty(self, initial_image, include_likelihood,
                                                num_steps, labels, model, include_layer, targets):
        images_list = []
        transformed_low, transformed_high = self.image_zero, self.image_one
        n = targets.shape[0]
        for label in labels:
            images = torch.zeros(n, 1, 28, 28)
            images += initial_image  # Use same initial image for each digit
            images_list.append(images)
            self.recover_image_batch(model, images, targets, num_steps, include_layer[label],
                                     label,
                                     include_likelihood)

        post_processed_images_list = []
        for images in images_list:
            # post_process_images(images)
            copied_images = images.clone().detach()
            # post_process_images(copied_images, mode='low_high', low=-0.5, high=2.0)
            imp.post_process_images(copied_images, mode='low_high',
                                    low=transformed_low,
                                    high=transformed_high)
            post_processed_images_list.append(copied_images)

        # One folder per digit, containing filtered and unfiltered images for that
        # digit
        plotter.generate_multi_plots_separate_digits(images_list,
                                                     post_processed_images_list, targets, labels)

        # One large image each (filtered and unfiltered) containing all digits,
        # all penalties
        plotter.generate_multi_plot_all_digits(images_list,
                                               post_processed_images_list, targets, labels)

        return images_list, post_processed_images_list

    # Load images_list from saved .pt files
    # Do post-processing only and plot
    def load_and_plot_images_varying_penalty(self, labels, targets):
        transformed_low, transformed_high = self.image_zero, self.image_one
        images_list = torch.load("images_list.pt")
        assert len(labels) == len(images_list)
        post_processed_images_list = []
        for images in images_list:
            copied_images = images.clone().detach()
            assert targets.shape[0] == copied_images.shape[0]
            imp.post_process_images(copied_images, mode='low_high',
                                    low=transformed_low,
                                    high=transformed_high)
            post_processed_images_list.append(copied_images)

        # One folder per digit, containing filtered and unfiltered images for that
        # digit
        # plotter.generate_multi_plots_separate_digits(images_list,
        #        post_processed_images_list, targets, labels)

        # One large image each (filtered and unfiltered) containing all digits,
        # all penalties
        plotter.generate_multi_plot_all_digits(images_list,
                                               post_processed_images_list, targets, labels)

        return images_list, post_processed_images_list

    def recover_and_plot_single_image(self, initial_image, digit, model, include_layer):
        label = "input only"
        targets = torch.tensor([digit])
        self.recover_image_batch(model, initial_image, targets, 2000, include_layer[label],
                                 label)
        plotter.show_image(initial_image[0][0])
        imp.post_process_images(initial_image, mode='low_high', low=-0.5, high=2.0)
        plotter.show_image(initial_image[0][0])

