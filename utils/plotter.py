import matplotlib
import torch

from datasets.dataset_helper_factory import DatasetHelperFactory

matplotlib.use('Agg') # For non-gui flow. Gets rid of DISPLAY bug in TkInter
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pathlib

import utils.mnist_helper as mh

# Useful globals
run_dir = '.'

def set_image_zero_one():
    global mnist_zero, mnist_one, mean, std
    #mnist_zero, mnist_one = mh.compute_mnist_transform_low_high()
    mnist_zero, mnist_one = DatasetHelperFactory.get().get_transformed_zero_one()
    mean, std = DatasetHelperFactory.get().get_mean_std_correct_dims(include_batch=False)
    print(mean, std)

def set_run_dir(some_dir):
    global run_dir
    run_dir = some_dir

# Gets image in the range [0, 1] by undoing the transformation done on the training dataset
def get_transformed_image(image):
    print(std)
    print(mean)
    print('Before transform: image min, max = ', torch.amin(image, dim=(1, 2)), torch.amax(image, dim=(1, 2)))
    image = image * std + mean
    print('After transform: image min, max = ', torch.amin(image, dim=(1, 2)), torch.amax(image, dim=(1, 2)))
    image = image.permute((1, 2, 0)) # matplotlib expects H x W x C
    return image

# vmin and vmax are ignore in case of RGB image
def plot_image_on_axis(ax, image, title, fig, vmin=None, vmax=None):
    shape = image.shape
    assert len(shape) == 2 or len(shape) == 3
    if len(shape) == 2:
        im = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    elif shape[2] == 1:
        im = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        # This is an RGB image of shape h x w x channel
        im = ax.imshow(image)

    ax.set_title(title)

    # Add colorbar for this image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


def plot_single_digit(image, digit, label, filtered):
    fig = plot.gcf()
    ax = plot.gca()
    title = "%d : %s" % (digit, label)
    (vmin, vmax) = (0., 1.) if filtered else (None, None)
    # We will first transform the image to the range (0, 1)
    image = get_transformed_image(image)
    plot_image_on_axis(ax, image, title, fig, vmin=vmin, vmax=vmax)
    filtered_str = "filtered" if filtered else "unfiltered"
    filename = f"{run_dir}/output/{digit}_{label}_{filtered_str}.jpg"
    print(filename)
    plot.savefig(filename)
    plot.close()

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
            image = images[j]
            ax = axes[i][j]
            title = "%d : %s" % (targets[j], labels[i])
            #plot_image_on_axis(ax, image, title, fig)
            # We will first transform the image to the range (0, 1)
            image = get_transformed_image(image)
            plot_image_on_axis(ax, image, title, fig, vmin=mnist_zero, vmax=mnist_one)

    plot.tight_layout(pad=2.)
    plot.savefig(filename)
    #plot.show()
    plot.clf()
    plot.rcParams.update({'font.size' : 10 })
    plot.close()

def generate_multi_plot_all_digits(images_list, post_processed_images_list, targets, labels):
    #filename = "./output/mean_0.5/10k/unfiltered_10k_varying_penalty.jpg"
    filename = f"{run_dir}/output/all_digits_unfiltered_varying_penalty.jpg"
    plot_multiple_images_varying_penalty(filename, images_list, targets,
            labels)

    #filename = "./output/mean_0.5/10k/filtered_10k_varying_penalty.jpg"
    #filename = "./output/mean_0.5/2k/filtered_2k_varying_penalty.jpg"
    # filename = f"{run_dir}/output/all_digits_filtered_varying_penalty.jpg"
    # plot_multiple_images_varying_penalty(filename, post_processed_images_list, targets,
    #         labels)

def generate_multi_plots_separate_digits(images_list,
        post_processed_images_list, targets, labels):
    for i in range(len(targets)):
        digit = targets[i]
        #filename = f"./output/mean_0.5/10k/{digit}/unfiltered_10k_varying_penalty.jpg"
        filename = f"{run_dir}/output/{digit}_unfiltered_varying_penalty.jpg"
        path = pathlib.Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_multiple_images_varying_penalty_single_digit(filename, images_list, targets,
                labels, i, filtered=False)

        #filename = f"./output/mean_0.5/10k/{digit}/filtered_10k_varying_penalty.jpg"
        # filename = f"{run_dir}/output/{digit}_filtered_varying_penalty.jpg"
        # plot_multiple_images_varying_penalty_single_digit(filename,
        #         post_processed_images_list, targets,
        #         labels, i, filtered=True)


def get_range_filtered(filtered):
    return (mnist_zero, mnist_one) if filtered else (None, None)

# 7 items to plot
# 3 rows, 3 cols
def plot_multiple_images_varying_penalty_single_digit(filename, images_list, targets,
        labels, index, filtered):
    num_images = len(images_list)
    assert index < len(targets)
    nrows = 3
    ncols = 3
    assert len(labels) == num_images
    plot.rcParams.update({'font.size' : 40 })
    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=(24, 24))
    vmin, vmax = get_range_filtered(filtered)
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            fig.delaxes(ax)
            continue
        images = images_list[i]
        image = images[index]
        # We will first transform the image to the range (0, 1)
        image = get_transformed_image(image)
        title = "%d : %s" % (targets[index], labels[i])
        #plot_image_on_axis(ax, image, title, fig, vmin=-0.5, vmax=2.0)
        #plot_image_on_axis(ax, image, title, fig, vmin=mnist_zero, vmax=mnist_one)
        plot_image_on_axis(ax, image, title, fig, vmin=mnist_zero, vmax=mnist_one)

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


