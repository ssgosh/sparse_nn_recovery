import torch
import torchvision.transforms

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
