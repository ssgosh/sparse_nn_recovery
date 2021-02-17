from __future__ import print_function

import argparse
import sys

import numpy as np
import torch

from utils import mnist_helper as mh
from utils import plotter
from utils import runs_helper as rh
# For experiment management
# import wandb
from utils.tensorboard_helper import TensorBoardHelper

np.set_printoptions(precision = 3)


def get_class(classname):
    return getattr(sys.modules[__name__], classname)


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
        help='L1 penalty lambda on each layer')
parser.add_argument('--digit', type=int, metavar='L',
        default=0, required=False,
        help='Which digit if single-digit is specified in mode')
parser.add_argument('--penalty-mode', type=str, default='input only', required=False,
        help='When mode is single-digit, which penalty mode should be used')
parser.add_argument('--disable-pgd', dest='use_pgd', action='store_false',
        default=True, required=False,
        help='Disable Projected Gradient Descent (clipping)')
parser.add_argument('--enable-pgd', dest='use_pgd', action='store_true',
        default=True, required=False,
        help='Enable Projected Gradient Descent (clipping)')

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
labels = list(include_layer.keys())
#labels = [ "input only", "all layers" ]
#labels = [ "input only", ]

config.include_layer = include_layer
config.labels = labels

# Run-specific information
#config.num_steps = 1000
config.include_likelihood = True
#config.lambd = 1. #0.1
#config.lambd_layers = [1., 1., 1.] #[0.1, 0.1, 0.1]
#config.lambd = 0.1
config.lambd_layers = 3 * [config.lambd] #[0.1, 0.1, 0.1]
#print(config.lambd_layers)
#sys.exit(1)

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
    label = config.penalty_mode
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

