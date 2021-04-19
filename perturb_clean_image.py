import math

import torch
import torch.nn.functional as F
import torchvision.transforms
from matplotlib import pyplot as plot

from datasets.dataset_helper import DatasetHelper
from datasets.dataset_helper_factory import DatasetHelperFactory

dataset_helper: DatasetHelper = DatasetHelperFactory.get('mnist', non_sparse=False)
dataset = dataset_helper.get_dataset(which='train', transform=torchvision.transforms.ToTensor())
print(dataset[0])
print(dataset[0][0].shape)
print(torch.max(dataset[0][0]).item(), torch.min(dataset[0][0]).item())

num = len(dataset)
idx = torch.randint(0, num, ()).item()
image, target = dataset[idx]

image = image.permute((1, 2, 0))
plot.imshow(image, cmap='gray')
# plot.show()

d = torch.load('ckpt/attack/images_list.pt')
sparse_attack_images = d['images'][1]
attack_targets = d['targets']

attack_image = sparse_attack_images[1]
attack_image = attack_image.permute((1, 2, 0))
attack_target = attack_targets[1]
plot.imshow(attack_image, cmap='gray')
# plot.show()

class Config:
    pass

config = Config()
dataset_helper.setup_config(config)
config.discriminator_model_file = 'ckpt/mnist_cnn.pt'
model = dataset_helper.get_model('max-entropy', device='cpu', config=config, load=True)

(mean,), (std,) = dataset_helper.get_mean_std()
print(mean, std)
for lambd in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]:
    perturbed_image = torch.clamp(image + lambd * attack_image, 0., 1.)
    batch = perturbed_image.unsqueeze(0)
    batch = batch.permute((0, 3, 1, 2))  # Channel dimension should be second
    batch = (batch - mean) / std
    output = model(batch)
    print(output)
    # probs = F.softmax(output)
    probs = torch.pow(math.e, output)
    print('lambda = ', lambd)
    print(probs)
    plot.imshow(perturbed_image, cmap='gray')
    plot.show()
