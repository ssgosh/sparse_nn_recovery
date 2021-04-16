import torch
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
plot.show()

d = torch.load('ckpt/attack/images_list.pt')
sparse_attack_images = d['images'][1]
attack_targets = d['targets']

attack_image = sparse_attack_images[0]
attack_image = attack_image.permute((1, 2, 0))
attack_target = attack_targets[0]
plot.imshow(attack_image, cmap='gray')
plot.show()

lambd = 0.1
perturbed_image = torch.clamp(image + lambd * attack_image, 0., 1.)
plot.imshow(perturbed_image, cmap='gray')
plot.show()
