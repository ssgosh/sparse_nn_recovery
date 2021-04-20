import math
import sys

import matplotlib
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms
from matplotlib import pyplot as plot
from torch.utils.data import Subset, DataLoader

from datasets.dataset_helper import DatasetHelper
from datasets.dataset_helper_factory import DatasetHelperFactory
from utils import plotter

#matplotlib.use('TkAgg')


# Fake config class
class Config:
    def __init__(self):
        self.device = 'cuda'


class Stats:
    def __init__(self):
        self.stats = {}

    def accumulate(self, probs, preds, d1, d2, alpha):
        sd = self.get_stats_dict(d1, d2, alpha)
        success = preds == d1
        failure = preds == d2
        #something_else = torch.logical_and(preds != d1, preds != d2)
        something_else = ((preds != d1) * (preds != d2)) > 0.

        attack_success = torch.sum(success).item()
        attack_failure = torch.sum(failure).item()
        attack_something_else = torch.sum(something_else).item()

        avg_prob_success = 0. if attack_success == 0. else torch.sum(probs[success]).item() / attack_success
        avg_prob_failure = 0. if attack_failure == 0. else torch.sum(probs[failure]).item() / attack_failure
        avg_prob_something_else = 0. if attack_something_else == 0. else torch.sum(probs[something_else]).item() / attack_something_else

        n = probs.shape[0]
        sd['frac_attack_class'] = attack_success / n
        sd['frac_actual_class'] = attack_failure / n
        sd['frac_other_class'] = attack_something_else / n

        sd['prob_attack_class'] = avg_prob_success
        sd['prob_actual_class'] = avg_prob_failure
        sd['prob_other_class'] = avg_prob_something_else

    def get_stats_dict(self, d1, d2, alpha):
        if d1 not in self.stats:
            self.stats[d1] = {}
        if d2 not in self.stats[d1]:
            self.stats[d1][d2] = {}
        if alpha not in self.stats[d1][d2]:
            self.stats[d1][d2][alpha] = {}
        return self.stats[d1][d2][alpha]

    def finalize(self):
        pass

    def dump(self, attack_probs):
        with open('sparse_attack_stats.p', 'wb') as f:
            pickle.dump(self.stats, f)
            pickle.dump(attack_probs, f)

        for d1 in self.stats:
            print(f'class of attack image : {d1}, prob = {attack_probs[d1]:.4f}')
            print('Fraction of dataset images classified as attack class, for different ground truth classes and various alpha')
            print('alpha\t', '\t'.join([f'{d2}' for d2 in self.stats[d1]]))
            # print('||')
            alphas = list(self.stats[d1][0].keys())
            for alpha in alphas:
                vals = [f"{self.stats[d1][d2][alpha]['frac_attack_class']:.2f}" for d2 in self.stats[d1]]
                vals.insert(0, str(alpha))
                print('\t'.join(vals))


class ImageAttack:
    def __init__(self):
        self.dataset_helper: DatasetHelper = DatasetHelperFactory.get('mnist', non_sparse=False)
        self.config = Config()
        self.dataset_helper.setup_config(self.config)
        plotter.set_image_zero_one()
        self.dataset = self.dataset_helper.get_dataset(which='train', transform=torchvision.transforms.ToTensor())
        self.mean, self.std = self.dataset_helper.get_mean_std_correct_dims(include_batch=False)
        #print(self.dataset[0])
        #print(self.dataset[0][0].shape)
        #print(torch.max(self.dataset[0][0]).item(), torch.min(self.dataset[0][0]).item())
        self.num = len(self.dataset)
        image_dict = torch.load('ckpt/attack/images_list.pt')
        self.sparse_attack_images = image_dict['images'][1].to(self.config.device)
        self.attack_targets = image_dict['targets'].to(self.config.device)

        self.config.discriminator_model_file = 'ckpt/mnist_cnn.pt'
        self.model = self.dataset_helper.get_model('max-entropy', device=self.config.device, config=self.config, load=True)
        self.model.train(False)

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
        assert d == attack_target
        attack_image = attack_image * self.std + self.mean
        epsilon = 1. / 256
        #print(f'Num less than epsilon ({epsilon}) = ', torch.sum(torch.logical_and(attack_image < epsilon, attack_image > 0.)).item())
        print(f'Num less than epsilon ({epsilon}) = ', torch.sum( ((attack_image < epsilon) * (attack_image > 0.)) > 0.).item())
        attack_image[attack_image < epsilon] = 0
        #print(f'Num less than epsilon ({epsilon}) = ', torch.sum(torch.logical_and(attack_image < epsilon, attack_image > 0.)).item())
        print(f'Num less than epsilon ({epsilon}) = ', torch.sum( ((attack_image < epsilon) * (attack_image > 0.)) > 0.).item())
        print(f'Sparsity = ', torch.sum(attack_image > 0.).item())
        if show:
            print('Attack image shape: ', attack_image.shape)
            print('Attack image min, max:', attack_image.min(), attack_image.max())
            plotter.plot_single_digit(attack_image, attack_target, 'Attack Image', filtered=True, show=True,
                                      transform=False)
        return attack_image, attack_target

    def attack_manual_check(self):
        mean, std = self.mean.unsqueeze(0), self.std.unsqueeze(0)
        image, target = self.choose_image(show=True)
        attack_image, attack_target = self.choose_attack_image(4, show=True)
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            perturbed_image = torch.clamp(image + alpha * attack_image, 0., 1.)
            print('Perturbed image shape: ', perturbed_image.shape)
            batch = perturbed_image.unsqueeze(0)
            batch = (batch - mean) / std
            output = self.model(batch)
            print(output)
            probs = torch.pow(math.e, output)[0]
            print('alphaa = ', alpha)
            print(probs)
            print('Perturbed image min, max:', perturbed_image.min(), perturbed_image.max())
            plotter.plot_single_digit(
                perturbed_image, target,
                f'Perturbed Image; alphaa = {alpha}, {target} : {probs[target]:.3f} -> {attack_target} : {probs[attack_target]:.3f}',
                filtered=True, show=True, transform=False)

    def sample_1000_images_per_class(self):
        with torch.no_grad():
            targets = self.dataset.targets.detach()
            # images = torch.tensor(self.dataset.images)
            dls = []
            n = 1
            for d in range(10):
                idx = (targets == d).nonzero().squeeze(-1)
                perm = torch.randperm(idx.shape[0])
                idx = idx[perm[0:n]]
                dls.append(DataLoader(Subset(self.dataset, idx), batch_size=n))
            return dls

    # Collect statistics about image attacks
    # What stats to collect?
    # For each attack image digit d1 in [0...9],
    # Sample 1000 dataset images each, with digit d2 in [0...9]
    # Choose alphaa in [0.1..1.0,...10.0]
    # For each pair (d1, d2, alphaa), collect the following:
    # Fraction of images classified as d1. (Pr >= 0.3)
    # Fraction of images classified as d2. (Pr >= 0.3)
    # Fraction of images classified as some other digit
    # Avg Pr[d1]
    # Avg Pr[d2]
    def attack_with_stats(self):
        stats = Stats()
        dls = im.sample_1000_images_per_class()
        mean, std = self.mean.unsqueeze(0), self.std.unsqueeze(0)
        attack_probs = []
        for d1 in range(10):
            with torch.no_grad():
                attack_image, attack_target = self.choose_attack_image(d1)
                attack_image = attack_image.unsqueeze(0)
                attack_out = self.model((attack_image - mean) / std)
                attack_prob = torch.pow(math.e, attack_out)[0][d1].item()
                attack_probs.append(attack_prob)
            for alpha1 in list(range(10)) + list(range(10, 110, 10)) + list(range(100, 1001, 100)) + [2000, 3000] :
                alpha = alpha1 / 10.
                for dl in dls:
                    for images, targets in dl:
                        with torch.no_grad():
                            images, targets = images.to(self.config.device), targets.to(self.config.device)
                            d2 = targets[0].item()
                            perturbed_images = torch.clamp(images + alpha * attack_image, 0., 1.)
                            # Transform images
                            perturbed_images = (perturbed_images - mean) / std
                            output = self.model(perturbed_images)
                            probs = torch.pow(math.e, output)
                            # Find predictions by taking argmax of probs along dim 1
                            preds = torch.argmax(probs, 1)
                            # Collect stats as noted earlier
                            stats.accumulate(probs, preds, d1, d2, alpha)
        stats.finalize()
        stats.dump(attack_probs)


im = ImageAttack()
im.attack_with_stats()
# dls = im.sample_1000_images_per_class()
# for dl in dls:
#     for image, target in dl:
#         print(image.shape, target.shape)
#         print(target[0])

# im.attack_manual_check()
# attack_manual_check()
