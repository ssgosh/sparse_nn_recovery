from __future__ import print_function

import argparse
import json
import sys
import os

import jsonpickle
import numpy as np
import torch

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from utils import plotter
from utils import runs_helper as rh
from datasets.dataset_helper_factory import DatasetHelperFactory
from utils.ckpt_saver import CkptSaver
from utils.gitutils import save_git_info
from utils.image_processor import save_grid_of_images
from utils.tensorboard_helper import TensorBoardHelper

from core.sparse_input_recoverer import SparseInputRecoverer

# noinspection PyUnresolvedReferences
from models.mnist_model import ExampleCNNNet
# noinspection PyUnresolvedReferences
from models.mnist_max_norm_mlp import MaxNormMLP
# noinspection PyUnresolvedReferences
from models.mnist_mlp import MLPNet3Layer

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


def add_main_script_arguments():
    parser = argparse.ArgumentParser(description='Recover images from a '
                                                 'discriminative model by gradient descent on input')
    parser.add_argument('--mode', type=str, default='all-digits', required=False,
                        help='Image recovery mode: "single-digit" or "all-digits" or "gen-dataset"')
    parser.add_argument('--dataset', type=str, default='mnist', required=False, choices=['mnist', 'cifar'],
                        help='Which dataset images to recover')
    parser.add_argument('--run-dir', type=str, default=None, required=False,
                        help='Directory inside which outputs and tensorboard logs will be saved')
    parser.add_argument('--run-suffix', type=str, default='', required=False,
                        help='Will be appended to the run directory provided')
    parser.add_argument('--discriminator-model-class', type=str, metavar='DMC',
                        default='ExampleCNNNet', required=False,
                        help='Discriminator model class')
    parser.add_argument('--discriminator-model-file', type=str, metavar='DMF',
                        default='ckpt/mnist_cnn.pt', required=False,
                        help='Discriminator model file')
    parser.add_argument('--digit', type=int, metavar='L',
                        default=0, required=False,
                        help='Which digit if single-digit is specified in mode')
    parser.add_argument('--dump-config', action='store_true',
                        default=False, required=False,
                        help='Print config json and exit')
    parser.add_argument('--dataset-len', type=int, metavar='L',
                        default=128, required=False,
                        help='How many images to generate in dataset mode')
    return parser


def setup_config(config):
    config.use_cuda = torch.cuda.is_available()
    config.device = torch.device("cuda" if config.use_cuda else "cpu")
    # This will change when we support multiple datasets
    dh = DatasetHelperFactory.get(config.dataset)
    dh.setup_config(config)
    SparseInputRecoverer.setup_default_config(config)
    # initial_image = torch.normal(mnist_zero, 0.01, (1, 1, 28, 28))
    if config.dump_config:
        json.dump(vars(config), sys.stdout, indent=2, sort_keys=True)
        sys.exit(0)

    return config, config.include_layer, config.labels, dh


def setup_everything(argv):
    parser = add_main_script_arguments()
    SparseInputRecoverer.add_command_line_arguments(parser)
    SparseInputDatasetRecoverer.add_command_line_arguments(parser)
    config = parser.parse_args(argv)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config, include_layer, labels, dh = setup_config(config)

    config.run_suffix = f'_{config.mode}_{config.dataset}'
    rh.setup_run_dir(config, 'image_runs')
    plotter.set_run_dir(config.run_dir)
    plotter.set_image_zero_one()

    # model = load_model(config)
    model = dh.get_model('max-entropy', device=config.device, config=config, load=True)

    tbh = TensorBoardHelper(config.run_dir)

    sparse_input_recoverer = SparseInputRecoverer(config, tbh, verbose=True)

    tbh.log_config_as_text(config)
    tbh.flush()
    return config, include_layer, labels, model, tbh, sparse_input_recoverer, dh


def main():
    config, include_layer, labels, model, tbh, sparse_input_recoverer, dh = setup_everything(sys.argv[1:])
    #initial_image = torch.randn(1, 1, 28, 28)
    initial_image = torch.randn( *( [1] + list(dh.get_each_entry_shape()) ) ).to(config.device)
    #images_list = torch.load("images_list.pt")
    #post_processed_images_list = torch.load("post_processed_images_list.pt")

    #generate_multi_plot_all_digits(images_list,
    #        post_processed_images_list, targets, labels)

    config_str = jsonpickle.encode(vars(config), indent=2)
    with open(f"{config.run_dir}/config.json" , 'w') as f:
        f.write(config_str)
    #save_git_info(f'{config.run_dir}/gitinfo.diff')
    os.system(f"python3.9 utils/gitutils.py {config.run_dir}/gitinfo.diff")
    if config.mode == 'all-digits':
        n = 10
        targets = torch.tensor(range(n), device=config.device)
        config.num_targets = n
        config.targets = targets
        # labels = ['no penalty', 'input only']
        labels = ['input only']
        images_list, post_processed_images_list = sparse_input_recoverer.recover_and_plot_images_varying_penalty(
            initial_image,
            include_likelihood=config.recovery_include_likelihood,
            num_steps=config.recovery_num_steps,
            labels=labels,
            model=model,
            include_layer=include_layer,
            targets=targets
        )

        torch.save({'images' : images_list, 'targets' : targets, 'labels' : labels}, f"{config.run_dir}/ckpt/images_list.pt")
        # torch.save(post_processed_images_list, f"{config.run_dir}/ckpt/post_processed_images_list.pt")
    elif config.mode == 'single-digit':
        n = 1
        targets = torch.tensor([config.digit], device=config.device)
        config.num_targets = n
        config.targets = targets
        label = config.recovery_penalty_mode
        recovered_image = sparse_input_recoverer.recover_and_plot_single_digit(
            initial_image, label, targets, include_layer=include_layer, model=model)
        torch.save(recovered_image, f"{config.run_dir}/ckpt/recovered_image.pt")
    elif config.mode == 'gen-dataset':
        ckpt_saver = CkptSaver(f"{config.run_dir}/ckpt")
        sparse_input_recoverer.tensorboard_logging = False
        # config.recovery_batch_size = 32
        # config.recovery_prune_low_prob = 32
        if config.recovery_batch_size > config.dataset_len:
            config.recovery_batch_size = config.dataset_len
        dataset_recoverer = SparseInputDatasetRecoverer(
            sparse_input_recoverer,
            model,
            num_recovery_steps=config.recovery_num_steps,
            batch_size=config.recovery_batch_size,
            sparsity_mode=config.recovery_penalty_mode,
            num_real_classes=dh.get_num_classes(),
            dataset_len=config.dataset_len,
            each_entry_shape=dh.get_each_entry_shape(),
            device=config.device, ckpt_saver=ckpt_saver, config=config)
        images, targets, probs = dataset_recoverer.recover_image_dataset(mode='train', dataset_epoch=0)
        torch.save({'images' : images, 'targets' : targets, 'probs' : probs, 'labels' : [config.recovery_penalty_mode]},
                   f"{config.run_dir}/ckpt/images_list.pt")
        save_grid_of_images(f"{config.run_dir}/output/samples.png", images, targets, dh)
        # dataset_recoverer = SparseInputDatasetRecoverer(sparse_input_recoverer, model, num_recovery_steps=kk,
        #                                                 batch_size=bs, sparsity_mode=config.recovery_penalty_mode,
        #                                                 num_real_classes=10, dataset_len=n,
        #                                                 each_entry_shape=(1, 28, 28), device='cpu',
        #                                                 ckpt_saver=ckpt_saver, config=config)

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

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
