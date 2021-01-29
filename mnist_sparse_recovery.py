from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from torchsummary import summary
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import numpy as np
import math
import pathlib

# For experiment management
import wandb

from mnist_model import ExampleCNNNet
from mnist_mlp import MLPNet3Layer

np.set_printoptions(precision = 3)

def get_class(classname):
    return getattr(sys.modules[__name__], classname)

def compute_mnist_transform_low_high():
    mean = 0.1307
    std = 0.3081
    transform = transforms.Normalize(mean, std)
    low = torch.zeros(1, 1, 1)
    high = low + 1
    print(torch.sum(low).item(), torch.sum(high).item())
    transformed_low = transform(low).item()
    transformed_high = transform(high).item()
    print(transformed_low, transformed_high)
    return transformed_low, transformed_high

#compute_mnist_transform_low_high()
#sys.exit(1)

def undo_transform(image):
    mean = 0.1307
    std = 0.3081
    return mean + image * std

def plot_image_on_axis(ax, image, title, fig, vmin=None, vmax=None):
    im = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title)

    # Add colorbar for this image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

# 1 col each for:
#
# no penalty
# input
# layer 1
# layer 2
# layer 3
# all but input
# all
#
# 7 rows, 10 cols
def plot_multiple_images_varying_penalty(filename, images_list, targets,
        labels):
    nrows = len(images_list)
    ncols = len(targets)
    assert len(labels) == nrows
    plot.rcParams.update({'font.size' : 40 })
    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=(80, 56))
    for i, images in enumerate(images_list):
        assert images.shape[0] == ncols
        for j in range(ncols):
            image = images[j][0]
            ax = axes[i][j]
            title = "%d : %s" % (targets[j], labels[i])
            plot_image_on_axis(ax, image, title, fig)

    plot.tight_layout(pad=2.)
    plot.savefig(filename)
    #plot.show()
    plot.clf()
    plot.rcParams.update({'font.size' : 10 })
    plot.close()

def generate_multi_plot_all_digits(images_list, post_processed_images_list, targets, labels):
    #filename = "./output/mean_0.5/10k/unfiltered_10k_varying_penalty.jpg"
    filename = "./output/all_digits_unfiltered_varying_penalty.jpg"
    plot_multiple_images_varying_penalty(filename, images_list, targets,
            labels)

    #filename = "./output/mean_0.5/10k/filtered_10k_varying_penalty.jpg"
    #filename = "./output/mean_0.5/2k/filtered_2k_varying_penalty.jpg"
    filename = "./output/all_digits_filtered_varying_penalty.jpg"
    plot_multiple_images_varying_penalty(filename, post_processed_images_list, targets,
            labels)

def generate_multi_plots_separate_digits(images_list,
        post_processed_images_list, targets, labels):
    for i in range(len(targets)):
        digit = targets[i]
        #filename = f"./output/mean_0.5/10k/{digit}/unfiltered_10k_varying_penalty.jpg"
        filename = f"./output/{digit}_unfiltered_varying_penalty.jpg"
        path = pathlib.Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_multiple_images_varying_penalty_single_digit(filename, images_list, targets,
                labels, i)

        #filename = f"./output/mean_0.5/10k/{digit}/filtered_10k_varying_penalty.jpg"
        filename = f"./output/{digit}_filtered_varying_penalty.jpg"
        plot_multiple_images_varying_penalty_single_digit(filename,
                post_processed_images_list, targets,
                labels, i)

# 7 items to plot
# 3 rows, 3 cols
def plot_multiple_images_varying_penalty_single_digit(filename, images_list, targets,
        labels, index):
    num_images = len(images_list)
    assert index < len(targets)
    nrows = 3
    ncols = 3
    assert len(labels) == num_images
    plot.rcParams.update({'font.size' : 40 })
    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=(24, 24))
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            fig.delaxes(ax)
            continue
        images = images_list[i]
        image = images[index][0]
        title = "%d : %s" % (targets[index], labels[i])
        #plot_image_on_axis(ax, image, title, fig, vmin=-0.5, vmax=2.0)
        plot_image_on_axis(ax, image, title, fig)#, vmin=-0.5, vmax=2.0)

    plot.tight_layout(pad=2.)
    plot.savefig(filename)
    plot.clf()
    plot.rcParams.update({'font.size' : 10 })
    # Close this or we're gonna have a bad time with OOM if
    # called from within ipython
    plot.close() 

# Plot images in a 3x4 grid
# All digits, 0-9
# deprecated
def plot_multiple_images(filename, original, images, targets):
    images.requires_grad = False
    fig, axes = plot.subplots(nrows=3, ncols=4, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        if idx >= 11:
            fig.delaxes(ax)
            continue
        if idx != 0:
            image = images[idx-1][0]
            title = "%d" % targets[idx-1]
        else:
            image = original
            title = "original"

        im = ax.imshow(image, cmap='gray')
        ax.set_title(title)

        # Add colorbar for this image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    plot.tight_layout(pad=0.)
    plot.savefig(filename)
    plot.clf()

def show_image(image):
    save_requires_grad = image.requires_grad
    image.requires_grad = False
    print("Image mean, std, min, max: ", image.mean().item(),
            image.std().item(),
            image.min().item(), image.max().item())
    #print("Initial image: ", torch.sum(image[0][0]))
    imshow(image, cmap='gray')
    plot.colorbar()
    #plot.draw()
    #plot.pause(0.0001)
    #plot.show()
    #imshow(undo_transform(image)[0][0], cmap='gray')
    plot.show()
    image.requires_grad = save_requires_grad

# high-pass and low-pass filter for images
def post_process_images(images, mode='mean_median', low=None, high=None):
    n = images.shape[0]
    channel = 0
    for idx in range(n):
        image = images[idx][channel]
        if mode == 'mean_median':
            mean = image.mean()
            median = image.median()
            low = (median + mean) / 2
        elif mode == 'low_high':
            assert low is not None or high is not None
        else:
            raise ValueError("Invalid value provided for mode %s" % mode)

        if low:
            image[image <= low] = low
        if high:
            image[image >= high] = high


# include_layer: boolean vector of whether to include a layer's l1 penalty
def recover_image(model, images, targets, num_steps, include_layer,
        include_likelihood=True):
    images.requires_grad = True
    optimizer = optim.Adam([images], lr=0.05)

    # lambda for input
    lambd = config.lambd
    #lambd = 0.01
    # lambda for each layer
    lambd_layers = config.lambd_layers #[0.1, 0.1, 0.1]
    #lambd_layers = [0.01, 0.01, 0.01]
    #lambd2 = 1.
    for i in range(1, num_steps+1):
        optimizer.zero_grad()
        output = model(images)
        if include_likelihood:
            nll_loss = F.nll_loss(output, targets)
        else:
            nll_loss = 0.

        # include l1 penalty only if it's given as true for that layer
        l1_loss = 0.
        if include_layer[0]:
            l1_loss = lambd * (torch.norm(images + 0.5, 1)
                    / torch.numel(images))

        l1_layers = 0.
        for include, lamb, l1 in zip(include_layer[1:], lambd_layers,
                model.all_l1):
            if include:
                l1_layers += lamb * l1

        #l1_layers = sum([ (lamb * l1) for lamb, l1 in zip(lambd_layers,
        #    model.all_l1) ])
        #l2_loss = lambd2 * (torch.norm(images, + 2) ** 2
        #        / torch.numel(images))

        loss = nll_loss + l1_loss + l1_layers
        loss.backward()
        print("Iter: ", i,", Loss: %.3f" % loss.item(),
                f"Prob of {targets[0]} %.3f" %
                pow(math.e, output[0][targets[0].item()].item()),
                "images median, mean, std, min, max: %.3f, %.3f, %.3f, %.3f, %.3f" % (
                images.median().item(), images.mean().item(), images.std().item(), images.min().item(),
                images.max().item()))
        optimizer.step()

    images.requires_grad = False

def recover_and_plot_single_digit():
    recover_image(model, images, targets, 10000, include_layer[label],
            include_likelihood=False)

def post_process_images_list(images_list):
    transformed_low, transformed_high = compute_mnist_transform_low_high()
    post_processed_images_list = []
    for images in images_list:
        #post_process_images(images)
        copied_images = images.clone().detach()
        #post_process_images(copied_images, mode='low_high', low=-0.5, high=2.0)
        post_process_images(copied_images, mode='low_high',
                low=transformed_low,
                high=transformed_high)
        post_processed_images_list.append(copied_images)

    return post_processed_images_list


# The main loop
#
# Uses recover_image() to perform sparse recovery of digit images from a model
def recover_and_plot_images_varying_penalty(initial_image, include_likelihood,
        num_steps):
    images_list = []
    transformed_low, transformed_high = compute_mnist_transform_low_high()
    for label in labels:
        images = torch.zeros(n, 1, 28, 28)
        images += initial_image  # Use same initial image for each digit
        images_list.append(images)
        recover_image(model, images, targets, num_steps, include_layer[label],
                include_likelihood)

    post_processed_images_list = []
    for images in images_list:
        #post_process_images(images)
        copied_images = images.clone().detach()
        #post_process_images(copied_images, mode='low_high', low=-0.5, high=2.0)
        post_process_images(copied_images, mode='low_high',
                low=transformed_low,
                high=transformed_high)
        post_processed_images_list.append(copied_images)

    # One folder per digit, containing filtered and unfiltered images for that
    # digit
    generate_multi_plots_separate_digits(images_list,
            post_processed_images_list, targets, labels)

    # One large image each (filtered and unfiltered) containing all digits,
    # all penalties
    generate_multi_plot_all_digits(images_list,
            post_processed_images_list, targets, labels)

    return images_list, post_processed_images_list


# Load images_list from saved .pt files
# Do post-processing only and plot
def load_and_plot_images_varying_penalty():
    transformed_low, transformed_high = compute_mnist_transform_low_high()
    images_list = torch.load("images_list.pt")
    assert len(labels) == len(images_list)
    post_processed_images_list = []
    for images in images_list:
        copied_images = images.clone().detach()
        assert targets.shape[0] == copied_images.shape[0]
        post_process_images(copied_images, mode='low_high',
                low=transformed_low,
                high=transformed_high)
        post_processed_images_list.append(copied_images)

    # One folder per digit, containing filtered and unfiltered images for that
    # digit
    generate_multi_plots_separate_digits(images_list,
            post_processed_images_list, targets, labels)

    # One large image each (filtered and unfiltered) containing all digits,
    # all penalties
    generate_multi_plot_all_digits(images_list,
            post_processed_images_list, targets, labels)

    return images_list, post_processed_images_list


def recover_and_plot_single_image(initial_image, digit):
    label = "input only"
    targets = torch.tensor([digit])
    recover_image(model, initial_image, targets, 2000, include_layer[label])
    show_image(initial_image[0][0])
    post_process_images(initial_image, mode='low_high', low=-0.5, high=2.0)
    show_image(initial_image[0][0])

def load_model(config):
    model_class = get_class(config.discriminator_model_class)
    model = model_class()
    #model_state_dict = torch.load('mnist_cnn.pt')
    model_state_dict = torch.load(config.discriminator_model_file)
    model.load_state_dict(model_state_dict)
    # XXX: Must set this in order for dropout to go away
    model.eval()
    return model

run = wandb.init(project='mnist_sparse_recovery')
config = wandb.config

config.discriminator_model_class = 'ExampleCNNNet'
config.discriminator_model_file = 'mnist_cnn.pt'

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

initial_image = torch.randn(1, 1, 28, 28)
n = 10
targets = torch.tensor(range(n))
include_layer = {
        "no penalty"    : [ False, False, False, False],
        "input only"    : [ True, False, False, False],
        "all layers"    : [ True, True, True, True],
        "layer 1 only"  : [ False, True, False, False],
        "layer 2 only"  : [ False, False, True, False],
        "layer 3 only"  : [ False, False, False, True],
        "all but input" : [ False, True, True, True],
        }
labels = list(include_layer.keys())

config.num_targets = n
config.targets = targets
config.include_layer = include_layer
config.labels = labels

# Run-specific information
config.num_steps = 100
config.include_likelihood = True
config.lambd = 0.1
config.lambd_layers = [0.1, 0.1, 0.1]

#labels.remove("no penalty")

#images_list = torch.load("images_list.pt")
#post_processed_images_list = torch.load("post_processed_images_list.pt")

#generate_multi_plot_all_digits(images_list,
#        post_processed_images_list, targets, labels)

images_list, post_processed_images_list = recover_and_plot_images_varying_penalty(initial_image,
        include_likelihood=config.include_likelihood, num_steps=config.num_steps)

torch.save(images_list, "images_list.pt")
torch.save(post_processed_images_list, "post_processed_images_list.pt")

#wandb.save(images_list)
#wandb.save(post_processed_images_list)

#load_and_plot_images_varying_penalty()

#recover_and_plot_single_image(initial_image, 0)
#initial_image = torch.randn(1, 1, 28, 28)
#recover_and_plot_single_image(initial_image, 1)
#initial_image = torch.randn(1, 1, 28, 28)
#recover_and_plot_single_image(initial_image, 4)

#plot_multiple_images('./output/mean_0.5/10k/unfiltered_10k_all_penalty.png', initial_image[0][0], images, targets)
#post_process_images(images)
#plot_multiple_images('./output/mean_0.5/10k/filtered_10k_all_penalty.png', initial_image[0][0], images, targets)

#images_list = [images]*7

