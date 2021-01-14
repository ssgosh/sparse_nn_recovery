from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import numpy as np
import math

from mnist_model import Net

np.set_printoptions(precision = 3)

def undo_transform(image):
    mean = 0.1307
    std = 0.3081
    return mean + image * std

def plot_image_on_axis(ax, image, title, fig):
    im = ax.imshow(image, cmap='gray')
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
    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=(20, 14))
    for i, images in enumerate(images_list):
        assert images.shape[0] == ncols
        for j in range(ncols):
            image = images[j][0]
            ax = axes[i][j]
            title = "%d : %s" % (targets[j], labels[i])
            plot_image_on_axis(ax, image, title, fig)

    plot.tight_layout(pad=0.)
    plot.savefig(filename)
    #plot.show()
    plot.clf()


# Plot images in a 3x4 grid
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

def post_process_images(images):
    n = images.shape[0]
    channel = 0
    for idx in range(n):
        image = images[idx][channel]
        mean = image.mean()
        median = image.median()
        image[image <= (median + mean) / 2] = (median + mean) / 2


def recover_image(model, images, targets, num_steps):
    images.requires_grad = True
    optimizer = optim.Adam([images], lr=0.05)

    # lambda for input
    lambd = 0.1
    # lambda for each layer
    lambd_layers = [0.1, 0.1, 0.1]
    #lambd2 = 1.
    for i in range(1, num_steps+1):
        optimizer.zero_grad()
        output = model(images)
        nll_loss = F.nll_loss(output, targets)
        l1_loss = lambd * (torch.norm(images + 0.5, 1)
                / torch.numel(images))
        l1_layers = sum([ (lamb * l1) for lamb, l1 in zip(lambd_layers,
            model.all_l1) ])
        #l2_loss = lambd2 * (torch.norm(images, + 2) ** 2
        #        / torch.numel(images))

        #loss = nll_loss + l1_loss
        #loss = nll_loss + l1_layers 
        loss = nll_loss + l1_loss + l1_layers
        #loss = l1_layers
        #loss = nll_loss
        loss.backward()
        print("Iter: ", i,", Loss: %.3f" % loss.item(),
                f"Prob of {targets[0]} %.3f" %
                pow(math.e, output[0][targets[0].item()].item()),
                "images median, mean, std, min, max: %.3f, %.3f, %.3f, %.3f, %.3f" % (
                images.median().item(), images.mean().item(), images.std().item(), images.min().item(),
                images.max().item()))
        optimizer.step()

    images.requires_grad = False


model = Net()
model_state_dict = torch.load('mnist_cnn.pt')
model.load_state_dict(model_state_dict)
#print(model)
#summary(model, (1, 28, 28))

initial_image = torch.randn(1, 1, 28, 28)
n = 10
images = torch.zeros(n, 1, 28, 28)
images += initial_image  # Use same initial image for each digit
targets = torch.tensor(range(n))
#show_image(images[0][0])
recover_image(model, images, targets, 2000)
#for idx in range(n):
#    show_image(images[idx][0])
#    post_process_images(images)
#    show_image(images[idx][0])

#plot_multiple_images('./output/mean_0.5/10k/unfiltered_10k_all_penalty.png', initial_image[0][0], images, targets)
#post_process_images(images)
#plot_multiple_images('./output/mean_0.5/10k/filtered_10k_all_penalty.png', initial_image[0][0], images, targets)

images_list = [images]*7
labels = ["no penalty", "input only", "layer 1 only", "layer 2 only",
        "layer 3 only", "all but input", "all layers"]
filename = "./output/mean_0.5/10k/unfiltered_10k_varying_penalty.jpg"
plot_multiple_images_varying_penalty(filename, images_list, targets,
        labels)

post_process_images(images)

filename = "./output/mean_0.5/10k/filtered_10k_varying_penalty.jpg"
plot_multiple_images_varying_penalty(filename, images_list, targets,
        labels)
