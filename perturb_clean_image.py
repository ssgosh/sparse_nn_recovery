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

dataset_helper: DatasetHelper = DatasetHelperFactory.get('mnist', non_sparse=False)
dataset = dataset_helper.get_dataset(which='train', transform=torchvision.transforms.ToTensor())
mean, std = dataset_helper.get_mean_std_correct_dims(include_batch=False)
print(dataset[0])
print(dataset[0][0].shape)
print(torch.max(dataset[0][0]).item(), torch.min(dataset[0][0]).item())

num = len(dataset)
idx = torch.randint(0, num, ()).item()
image, target = dataset[idx]

# Setup plotter
plotter.set_image_zero_one()
#image = image.permute((1, 2, 0))
print('Dataset image shape: ', image.shape)
print('Attack image min, max:', image.min(), image.max())
plotter.plot_single_digit(image, target, 'Plain Dataset Image', filtered=True, show=True, transform=False)
#plot.imshow(image, cmap='gray')
# plot.show()

d = torch.load('ckpt/attack/images_list.pt')
sparse_attack_images = d['images'][1]
attack_targets = d['targets']

d = 5
attack_image = sparse_attack_images[d]
# attack_image = attack_image.permute((1, 2, 0))
attack_target = attack_targets[d]
attack_image = attack_image * std + mean
# plot.imshow(attack_image, cmap='gray')
print('Attack image shape: ', attack_image.shape)
print('Attack image min, max:', attack_image.min(), attack_image.max())
plotter.plot_single_digit(attack_image, attack_target, 'Attack Image', filtered=True, show=True, transform=False)
# sys.exit(0)
# plot.show()

class Config:
    def __init__(self):
        self.device = 'cpu'

config = Config()
dataset_helper.setup_config(config)
config.discriminator_model_file = 'ckpt/mnist_cnn.pt'
model = dataset_helper.get_model('max-entropy', device='cpu', config=config, load=True)

(mean,), (std,) = dataset_helper.get_mean_std()
print(mean, std)
for lambd in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
    perturbed_image = torch.clamp(image + lambd * attack_image, 0., 1.)
    print('Perturbed image shape: ', perturbed_image.shape)
    batch = perturbed_image.unsqueeze(0)
    #batch = batch.permute((0, 3, 1, 2))  # Channel dimension should be second
    batch = (batch - mean) / std
    output = model(batch)
    print(output)
    # probs = F.softmax(output)
    probs = torch.pow(math.e, output)[0]
    print('lambda = ', lambd)
    print(probs)
    print('Perturbed image min, max:', perturbed_image.min(), perturbed_image.max())
    plotter.plot_single_digit(
        perturbed_image, target,
        f'Perturbed Image; lambda = {lambd}, {target} : {probs[target]:.3f} -> {attack_target} : {probs[attack_target]:.3f}',
        filtered=True, show=True, transform=False)
    # plot.imshow(perturbed_image, cmap='gray')
    # plot.show()
