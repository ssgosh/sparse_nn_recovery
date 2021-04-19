import math
import sys

import matplotlib
import torch
import torch.nn.functional as F
import torchvision.transforms
from matplotlib import pyplot as plot

from datasets.dataset_helper import DatasetHelper
from datasets.dataset_helper_factory import DatasetHelperFactory
from utils import plotter
matplotlib.use('TkAgg')

# Fake config class
class Config:
    def __init__(self):
        self.device = 'cpu'

class ImageAttack:
    def __init__(self):
        self.dataset_helper: DatasetHelper = DatasetHelperFactory.get('mnist', non_sparse=False)
        plotter.set_image_zero_one()
        self.dataset = self.dataset_helper.get_dataset(which='train', transform=torchvision.transforms.ToTensor())
        self.mean, self.std = self.dataset_helper.get_mean_std_correct_dims(include_batch=False)
        print(self.dataset[0])
        print(self.dataset[0][0].shape)
        print(torch.max(self.dataset[0][0]).item(), torch.min(self.dataset[0][0]).item())
        self.num = len(self.dataset)
        image_dict = torch.load('ckpt/attack/images_list.pt')
        self.sparse_attack_images = image_dict['images'][1]
        self.attack_targets = image_dict['targets']

        self.config = Config()
        self.dataset_helper.setup_config(self.config)
        self.config.discriminator_model_file = 'ckpt/mnist_cnn.pt'
        self.model = self.dataset_helper.get_model('max-entropy', device='cpu', config=self.config, load=True)

    def choose_image(self, show=False):
        idx = torch.randint(0, self.num, ()).item()
        image, target = self.dataset[idx]
        if show:
            print('Dataset image shape: ', image.shape)
            print('Attack image min, max:', image.min(), image.max())
            plotter.plot_single_digit(image, target, 'Plain Dataset Image', filtered=True, show=True, transform=False)
        return image, target

    def choose_attack_image(self, d, show=False):
        # d = 5
        attack_image = self.sparse_attack_images[d]
        # attack_image = attack_image.permute((1, 2, 0))
        attack_target = self.attack_targets[d]
        attack_image = attack_image * self.std + self.mean
        if show:
            print('Attack image shape: ', attack_image.shape)
            print('Attack image min, max:', attack_image.min(), attack_image.max())
            plotter.plot_single_digit(attack_image, attack_target, 'Attack Image', filtered=True, show=True, transform=False)
        return attack_image, attack_target

    def attack_manual_check(self):
        mean, std = self.mean.unsqueeze(0), self.std.unsqueeze(0)
        image, target = self.choose_image(show=True)
        attack_image, attack_target = self.choose_attack_image(4, show=True)
        for lambd in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            perturbed_image = torch.clamp(image + lambd * attack_image, 0., 1.)
            print('Perturbed image shape: ', perturbed_image.shape)
            batch = perturbed_image.unsqueeze(0)
            batch = (batch - mean) / std
            output = self.model(batch)
            print(output)
            probs = torch.pow(math.e, output)[0]
            print('lambda = ', lambd)
            print(probs)
            print('Perturbed image min, max:', perturbed_image.min(), perturbed_image.max())
            plotter.plot_single_digit(
                perturbed_image, target,
                f'Perturbed Image; lambda = {lambd}, {target} : {probs[target]:.3f} -> {attack_target} : {probs[attack_target]:.3f}',
                filtered=True, show=True, transform=False)

    # Collect statistics about image attacks
    # What stats to collect?
    # For each attack image digit in [0...9],
    # Sample 1000 dataset images each.
    def attack_with_stats(self):
        pass


im = ImageAttack()
im.attack_manual_check()
# attack_manual_check()
