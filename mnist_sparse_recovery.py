from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

from torchsummary import summary
import numpy as np
import math

# For experiment management
#import wandb
from utils.tensorboard_helper import TensorBoardHelper

from utils import image_processor as imp
from utils import mnist_helper as mh
from utils import plotter
from utils import metrics_helper as mth
from utils import runs_helper as rh

from models.mnist_model import ExampleCNNNet
from models.mnist_mlp import MLPNet3Layer

np.set_printoptions(precision = 3)


def get_class(classname):
    return getattr(sys.modules[__name__], classname)


# Clip the pixels to between (mnist_zero, mnist_one)
def clip_if_needed(images):
    mnist_zero, mnist_one = mh.compute_mnist_transform_low_high()
    with torch.no_grad():
        torch.clip(images, mnist_zero, mnist_one, out=images)


# include_layer: boolean vector of whether to include a layer's l1 penalty
def recover_image(model, images, targets, num_steps, include_layer, label,
        include_likelihood=True):
    mnist_zero, mnist_one = mh.compute_mnist_transform_low_high()
    images.requires_grad = True
    optimizer = optim.Adam([images], lr=0.5)

    # lambda for input
    lambd = config.lambd
    #lambd = 0.01
    # lambda for each layer
    lambd_layers = config.lambd_layers #[0.1, 0.1, 0.1]
    #lambd_layers = [0.01, 0.01, 0.01]
    #lambd2 = 1.
    losses = {}
    probs = {}
    sparsity = {}
    for i in range(1, num_steps+1):
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
            l1_loss = lambd * (torch.norm(images - mnist_zero, 1)
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
        #l1_layers = sum([ (lamb * l1) for lamb, l1 in zip(lambd_layers,
        #    model.all_l1) ])
        #l2_loss = lambd2 * (torch.norm(images, + 2) ** 2
        #        / torch.numel(images))

        loss = nll_loss + l1_loss + l1_layers
        loss.backward()

        # Do step before computation of metrics which will change on backprop
        optimizer.step()
        clip_if_needed(images)

        losses[f"total_loss"] = loss.item()
        for idx, tgt in enumerate(targets):
            prob = pow(math.e, output[idx][tgt.item()].item())
            #print(prob)
            probs[f"1-class_{tgt}/prob"] = prob
        mth.compute_sparsities(images, model, targets, sparsity)
        print("Iter: ", i,", Loss: %.3f" % loss.item(),
                f"Prob of {targets[0]} %.3f" %
                pow(math.e, output[0][targets[0].item()].item()),
                "images median, mean, std, min, max: %.3f, %.3f, %.3f, %.3f, %.3f" % (
                images.median().item(), images.mean().item(), images.std().item(), images.min().item(),
                images.max().item()))

        # Do tensorboard things
        tbh.add_tensorboard_stuff(label, model, images, losses, probs,
                sparsity, i)

    images.requires_grad = False


# Single digit, single label
def recover_and_plot_single_digit(initial_image, label, targets):
    recover_image(model, initial_image, targets, config.num_steps, include_layer[label], label,
            include_likelihood=True)
    plotter.plot_single_digit(initial_image.detach()[0][0], targets[0], label)
    return initial_image


# The main loop
#
# Uses recover_image() to perform sparse recovery of digit images from a model
def recover_and_plot_images_varying_penalty(initial_image, include_likelihood,
        num_steps):
    images_list = []
    transformed_low, transformed_high = mh.compute_mnist_transform_low_high()
    for label in labels:
        images = torch.zeros(n, 1, 28, 28)
        images += initial_image  # Use same initial image for each digit
        images_list.append(images)
        recover_image(model, images, targets, num_steps, include_layer[label],
                label,
                include_likelihood)

    post_processed_images_list = []
    for images in images_list:
        #post_process_images(images)
        copied_images = images.clone().detach()
        #post_process_images(copied_images, mode='low_high', low=-0.5, high=2.0)
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
def load_and_plot_images_varying_penalty():
    transformed_low, transformed_high = mh.compute_mnist_transform_low_high()
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
    #plotter.generate_multi_plots_separate_digits(images_list,
    #        post_processed_images_list, targets, labels)

    # One large image each (filtered and unfiltered) containing all digits,
    # all penalties
    plotter.generate_multi_plot_all_digits(images_list,
            post_processed_images_list, targets, labels)

    return images_list, post_processed_images_list


def recover_and_plot_single_image(initial_image, digit):
    label = "input only"
    targets = torch.tensor([digit])
    recover_image(model, initial_image, targets, 2000, include_layer[label],
            label)
    plotter.show_image(initial_image[0][0])
    imp.post_process_images(initial_image, mode='low_high', low=-0.5, high=2.0)
    plotter.show_image(initial_image[0][0])

def load_model(config):
    model_class = get_class(config.discriminator_model_class)
    model = model_class(num_classes=20)
    #model_state_dict = torch.load('mnist_cnn.pt')
    model_state_dict = torch.load(config.discriminator_model_file)
    model.load_state_dict(model_state_dict)
    # XXX: Must set this in order for dropout to go away
    model.eval()
    return model


def get_config_dict(config):
    return vars(config)


parser = argparse.ArgumentParser(description='Recover images from a '
        'discriminative model by gradient descent on input')
parser.add_argument('--mode', type=str, default='all-digits', required=False,
        help='Image recovery mode: "single-digit" or "all-digits"')
parser.add_argument('--num-steps', type=int, default=1000, required=False,
        help='Number of steps of gradient descent for image generation')
parser.add_argument('--run-dir', type=str, default=None, required=False,
        help='Directory inside which outputs and tensorboard logs will be saved')
parser.add_argument('--run-suffix', type=str, default='', required=False,
        help='Directory inside which outputs and tensorboard logs will be saved')
parser.add_argument('--discriminator-model-class', type=str, metavar='DMC',
        default='ExampleCNNNet', required=False,
        help='Discriminator model class')
parser.add_argument('--discriminator-model-file', type=str, metavar='DMF',
        default='ckpt/mnist_cnn.pt', required=False,
        help='Discriminator model file')
parser.add_argument('--lambd', type=float, metavar='L',
        default=0.1, required=False,
        help='L1 penalty lambda on input layer')
parser.add_argument('--digit', type=int, metavar='L',
        default=0, required=False,
        help='Which digit if single-digit is specified in mode')

config = parser.parse_args()

rh.setup_run_dir(config, 'image_runs')
plotter.set_run_dir(config.run_dir)

#run = wandb.init(project='mnist_sparse_recovery')
#config = wandb.config

#config.discriminator_model_class = 'ExampleCNNNet'
#config.discriminator_model_class = 'MLPNet3Layer'
#config.discriminator_model_file = 'ckpt/mnist_cnn_adv_normal_init.pt'
#config.discriminator_model_file = f'ckpt/mnist_cnn.pt'
#config.discriminator_model_file = 'ckpt/mnist_mlp_3layer_adv_normal_init.pt'

# Alternate model configuration
#wandb.config.discriminator_model_class = 'MLPNet3Layer'
#wandb.config.discriminator_model_file = 'mnist_mlp_3layer.pt'

model = load_model(config)

#model = Net()
#model = MLPNet3Layer()
#model_state_dict = torch.load('mnist_cnn.pt')
#model_state_dict = torch.load('mnist_mlp_3layer.pt')
#model.load_state_dict(model_state_dict)
#print(model)
#summary(model, (1, 28, 28))

mnist_zero, mnist_one = mh.compute_mnist_transform_low_high()
initial_image = torch.randn(1, 1, 28, 28)
#initial_image = torch.normal(mnist_zero, 0.01, (1, 1, 28, 28))
include_layer = {
        "no penalty"    : [ False, False, False, False],
        "input only"    : [ True, False, False, False],
        "all layers"    : [ True, True, True, True],
        "layer 1 only"  : [ False, True, False, False],
        "layer 2 only"  : [ False, False, True, False],
        "layer 3 only"  : [ False, False, False, True],
        "all but input" : [ False, True, True, True],
        }
#labels = list(include_layer.keys())
#labels = [ "input only", "all layers" ]
labels = [ "input only", ]

config.include_layer = include_layer
config.labels = labels

# Run-specific information
#config.num_steps = 1000
config.include_likelihood = True
#config.lambd = 1. #0.1
#config.lambd_layers = [1., 1., 1.] #[0.1, 0.1, 0.1]
#config.lambd = 0.1
config.lambd_layers = [0.1, 0.1, 0.1]

tbh = TensorBoardHelper(config.run_dir)

#labels.remove("no penalty")

#images_list = torch.load("images_list.pt")
#post_processed_images_list = torch.load("post_processed_images_list.pt")

#generate_multi_plot_all_digits(images_list,
#        post_processed_images_list, targets, labels)

if config.mode == 'all-digits':
    n = 10
    targets = torch.tensor(range(n))
    config.num_targets = n
    config.targets = targets
    images_list, post_processed_images_list = recover_and_plot_images_varying_penalty(initial_image,
            include_likelihood=config.include_likelihood, num_steps=config.num_steps)

    torch.save(images_list, f"{config.run_dir}/ckpt/images_list.pt")
    torch.save(post_processed_images_list, f"{config.run_dir}/ckpt/post_processed_images_list.pt")
elif config.mode == 'single-digit':
    n = 1
    targets = torch.tensor([config.digit])
    config.num_targets = n
    config.targets = targets
    label = labels[0]
    recovered_image = recover_and_plot_single_digit(initial_image, label, targets)
    torch.save(recovered_image, f"{config.run_dir}/ckpt/recovered_image.pt")
else:
    raise ValueError("Invalid mode %s" % config.mode)

#load_and_plot_images_varying_penalty()

#wandb.save("output/*")

#recover_and_plot_single_image(initial_image, 0)
#initial_image = torch.randn(1, 1, 28, 28)
#recover_and_plot_single_image(initial_image, 1)
#initial_image = torch.randn(1, 1, 28, 28)
#recover_and_plot_single_image(initial_image, 4)

#plotter.plot_multiple_images('./output/mean_0.5/10k/unfiltered_10k_all_penalty.png', initial_image[0][0], images, targets)
#post_process_images(images)
#plotter.plot_multiple_images('./output/mean_0.5/10k/filtered_10k_all_penalty.png', initial_image[0][0], images, targets)

#images_list = [images]*7

