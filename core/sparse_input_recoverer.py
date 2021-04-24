from __future__ import print_function

import argparse
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from datasets.dataset_helper_factory import DatasetHelperFactory
from utils import image_processor as imp
from utils import plotter
from utils.metrics_helper import MetricsHelper
from utils.model_context_manager import model_eval_no_grad, images_require_grad
from utils.torchutils import compute_probs_tensor, clip_tensor_range


class SparseInputRecoverer:

    # Dictionary of string penalty modes to array of boolean values,
    # indicating whether penalty is on/off for the corresponding layer in that mode
    include_layer_map = {
        "no penalty": [False, False, False, False],
        "input only": [True, False, False, False],
        "all layers": [True, True, True, True],
        "layer 1 only": [False, True, False, False],
        "layer 2 only": [False, False, True, False],
        "layer 3 only": [False, False, False, True],
        "all but input": [False, True, True, True],
    }

    # All available penalty modes
    all_penalty_modes = list(include_layer_map.keys())

    @staticmethod
    def add_command_line_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--recovery-num-steps', type=int, default=1000, required=False, metavar='N',
                            help='Number of steps of gradient descent for image generation')
        parser.add_argument('--recovery-lr', type=float, metavar='LR',
                            default=0.5, required=False,
                            help='Learning rate for sparse recovery')
        parser.add_argument('--recovery-lambd', type=float, metavar='L',
                            default=0.1, required=False,
                            help='L1 penalty lambda on each layer')
        parser.add_argument('--recovery-penalty-mode', type=str, default='input only', required=False, metavar='PM',
                            help='When mode is single-digit, which penalty mode should be used')
        parser.add_argument('--recovery-disable-pgd', dest='recovery_use_pgd', action='store_false',
                            default=True, required=False,
                            help='Disable Projected Gradient Descent (clipping)')
        parser.add_argument('--recovery-enable-pgd', dest='recovery_use_pgd', action='store_true',
                            default=True, required=False,
                            help='Enable Projected Gradient Descent (clipping)')

    @staticmethod
    def setup_default_config(config):
        config.include_layer = SparseInputRecoverer.include_layer_map
        config.labels = SparseInputRecoverer.all_penalty_modes
        config.recovery_include_likelihood = True
        config.recovery_lambd_layers = 3 * [config.recovery_lambd]  # [0.1, 0.1, 0.1]

    def __init__(self, config, tbh, verbose=False):
        """

        :type tbh: TensorBoardHelper
        :type verbose: bool
        """
        self.recovery_lambd = config.recovery_lambd
        self.recovery_lambd_layers = config.recovery_lambd_layers
        self.verbose = verbose
        self.config = config
        self.device = config.device
        self.tbh = tbh
        # Following are either floats or list of floats
        self.image_zero = config.image_zero
        self.image_one = config.image_one
        # Following are of shape [1, c, 1, 1], where c is the number of channels of the dataset
        self.batched_image_zero = DatasetHelperFactory.get().get_zero_correct_dims()
        self.batched_image_one = DatasetHelperFactory.get().get_one_correct_dims()
        self.batched_epsilon = DatasetHelperFactory.get().get_batched_epsilon()

        self.metrics_helper = MetricsHelper.get() # MetricsHelper(self.image_zero, self.image_one)
        # 'all' : both images and stats
        # 'none' : disable tensorboard logging
        # 'stats_only' : log only stats, no images
        self.tensorboard_logging = 'all'
        self.tensorboard_label = None

        # out_fn is either F.log_softmax or just identity depending on what the model does internally

    # Clip the pixels to between (mnist_zero, mnist_one)
    def clip_if_needed(self, images):
        if self.config.recovery_use_pgd:
            with torch.no_grad():
                #torch.clip(images, self.image_zero, self.image_one, out=images)
                #torch.clamp(images, self.image_zero, self.image_one, out=images)
                clip_tensor_range(images, self.batched_image_zero, self.batched_image_one, out=images)
                # Get eps1 in the same shape as images via broadcasting addition to a zero tensor
                # with the same shape as images
                eps1 = torch.zeros_like(images) + self.batched_epsilon
                # The index "images < eps1" is now valid for both: the tensor images and eps1, since they're
                # the same shape.
                images[images < eps1] = eps1[images < eps1]

    # include_layer: boolean vector of whether to include a layer's l1 penalty
    def recover_image_batch(self, model, images, targets, num_steps, include_layer, penalty_mode,
                            include_likelihood=True, batch_idx=0):
        with model_eval_no_grad(model), images_require_grad(images):
            return self.recover_image_batch_internal(model, images, targets, num_steps, include_layer, penalty_mode,
                                              include_likelihood, batch_idx)

    def recover_image_batch_internal(self, model, images, targets, num_steps, include_layer, penalty_mode,
                                     include_likelihood, batch_idx):
        optimizer = optim.Adam([images], lr=self.config.recovery_lr)

        tb_log = self.tensorboard_logging != 'none'
        tb_add_images = self.tensorboard_logging == 'all'
        tb_label = penalty_mode if self.tensorboard_label is None else self.tensorboard_label
        start = num_steps * batch_idx + 1
        for i in range(start, start + num_steps):
            optimizer.zero_grad()
            loss, losses, output, probs, sparsity = self.forward(model, images, targets, include_layer,
                                                                 include_likelihood)
            loss.backward()
            # step is done after metrics computations.
            # Hence metrics are for the batch as they came in, not as they went out
            optimizer.step()
            self.clip_if_needed(images)

            if self.verbose and i % 100 == 0:
                print("Iter: ", i, ", Loss: %.3f" % loss.item(),
                      f"Prob of {targets[0]} %.3f" %
                      pow(math.e, output[0][targets[0].item()].item()),
                      "images median, mean, std, min, max: %.3f, %.3f, %.3f, %.3f, %.3f" % (
                          images.median().item(), images.mean().item(), images.std().item(), images.min().item(),
                          images.max().item()))

            if tb_log:
                # Do tensorboard things
                self.tbh.add_tensorboard_stuff(tb_label, images, losses, probs,
                                               sparsity, i, add_images=tb_add_images)

        # Probabilities tensor computation after optimization is done
        # 1-d vector of length batch size, containing the probability of the target class
        with torch.no_grad():
            return compute_probs_tensor(output, targets)[1]


    def forward(self, model, images, targets, include_layer, include_likelihood):
        # lambda for input
        #lambd = self.config.recovery_lambd
        lambd = self.recovery_lambd
        # lambd = 0.01
        # lambda for each layer
        #lambd_layers = self.config.recovery_lambd_layers  # [0.1, 0.1, 0.1]
        lambd_layers = self.recovery_lambd_layers  # [0.1, 0.1, 0.1]
        # lambd_layers = [0.01, 0.01, 0.01]
        # lambd2 = 1.
        losses = {}
        probs = {}
        sparsity = {}
        output = model(images)
        if include_likelihood:
            nll_loss = F.nll_loss(output, targets)
        else:
            nll_loss = torch.tensor(0., device=self.device)
        losses[f"nll_loss"] = nll_loss.item()
        # include l1 penalty only if it's given as true for that layer
        l1_loss = torch.tensor(0., device=self.device)
        if include_layer[0]:
            l1_loss = lambd * (torch.norm(images - self.batched_image_zero, 1)
                               / torch.numel(images))
        losses[f"input_l1_loss"] = l1_loss.item()
        l1_layers = torch.tensor(0., device=self.device)
        for idx, (include, lamb, l1) in enumerate(zip(include_layer[1:], lambd_layers,
                                                      model.all_l1)):
            if include:
                layer_loss = lamb * l1
                l1_layers += layer_loss
                losses[f"layer_{idx}_l1_loss"] = layer_loss.item()
        losses[f"hidden_layers_l1_loss"] = l1_layers.item()
        losses[f"all_layers_l1_loss"] = l1_loss.item() + l1_layers.item()
        loss = nll_loss + l1_loss + l1_layers
        losses[f"total_loss"] = loss.item()
        self.metrics_helper.compute_probs(output, probs, targets)
        self.metrics_helper.compute_sparsities(images, model, targets, sparsity)

        return loss, losses, output, probs, sparsity

    # Single digit, single label
    def recover_and_plot_single_digit(self, initial_image, label, targets, include_layer, model):
        self.recover_image_batch(model, initial_image, targets, self.config.recovery_num_steps, include_layer[label], label,
                                 include_likelihood=True)
        plotter.plot_single_digit(initial_image.detach()[0], targets[0], label,
                                  filtered=False)
        plotter.plot_single_digit(initial_image.detach()[0], targets[0], label,
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
            images = torch.zeros([n] + list(initial_image.shape[1:]), device=self.device)
            images += initial_image  # Use same initial image for each digit
            images_list.append(images)
            self.recover_image_batch(model, images, targets, num_steps, include_layer[label],
                                     label,
                                     include_likelihood)

        post_processed_images_list = []
        # for images in images_list:
        #     # post_process_images(images)
        #     copied_images = images.clone().detach()
        #     # post_process_images(copied_images, mode='low_high', low=-0.5, high=2.0)
        #     imp.post_process_images(copied_images, mode='low_high',
        #                             low=transformed_low,
        #                             high=transformed_high)
        #     post_processed_images_list.append(copied_images)

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
        targets = torch.tensor([digit], device=self.device)
        self.recover_image_batch(model, initial_image, targets, 2000, include_layer[label],
                                 label)
        plotter.show_image(initial_image[0][0])
        imp.post_process_images(initial_image, mode='low_high', low=-0.5, high=2.0)
        plotter.show_image(initial_image[0][0])

    def anneal_lambda(self, lambda_annealing_factor):
        before = self.recovery_lambd
        self.recovery_lambd *= lambda_annealing_factor
        self.recovery_lambd_layers = 3 * [self.recovery_lambd]
        print(f"Annealed lambda to {self.recovery_lambd} from {before}")
