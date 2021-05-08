import pickle
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from datasets.non_sparse_normalization_mixin import NonSparseNormalizationMixin


# Abstraction for Dataset-specific functionality, such as transformations,
# values of transformed zero and one pixel values. Also provides a singleton for the datasethelper
# Used in the main scripts.
from utils.torchutils import get_cross


class DatasetHelper(ABC, NonSparseNormalizationMixin):

    def __init__(self, name, subset, non_sparse):
        self.name = name
        self.subset = subset
        self.non_sparse = non_sparse
        self.device = 'cpu'     # Device is cpu by default

    def get_dataset(self, which='train', transform=None) -> Dataset:
        """
        Return torch.utils.data.Dataset object for train/test/valid split with the given transform

        :param which:
        :param transform:
        :return:
        """
        train = (which == 'train')
        if transform == 'train' : transform = self.get_train_transform()
        elif transform == 'test' : transform = self.get_test_transform()
        elif transform == 'without_transform' : transform = self.get_without_transform()
        path = './data'
        if not self.subset : return self.get_dataset_(path, which, transform)

        # train/test splits were created from original train. Hence train=True in the following call.
        full_train_data = self.get_dataset_(path, which='train', transform=transform)
        fname = 'train' if train else 'test'
        path = f'./data/{self.name}/idx/{fname}.p'
        with open(path, 'rb') as f:
            idx = pickle.load(f)
        return Subset(full_train_data, idx)

    @abstractmethod
    def get_dataset_(self, path, which, transform):
        pass

    def get_transformed_zero_one(self):
        return self.get_transformed_zero_one_mixin()

    def setup_config(self, config):
        zero, one = self.get_transformed_zero_one()
        config.image_zero = zero
        config.image_one = one
        self.device = config.device
        self.update_config(config)

    # XXX: Rename method to get_num_real_classes and update usage.
    @abstractmethod
    def get_num_classes(self):
        pass

    # No need to be abstractmethod
    def get_num_real_fake_classes(self):
        # XXX: Change this to get_num_real_classes()
        return 2 * self.get_num_classes()

    # XXX: This should use adversarial_classification_mode and either return only real or real + fake classesj
    def get_num_total_classes(self):
        return self.get_num_classes()

    @abstractmethod
    def get_each_entry_shape(self):
        pass

    def get_zero_correct_dims(self, include_batch=True):
        zero, one = self.get_transformed_zero_one()
        return self.get_correct_dims(zero, include_batch, self.device)

    def get_one_correct_dims(self, include_batch=True): 
        zero, one = self.get_transformed_zero_one()
        return self.get_correct_dims(one, include_batch, self.device)

    # Returns a tensor of shape [1, c, 1, 1] or [c, 1, 1], where c = len(val) or 1 if val is a number
    def get_correct_dims(self, val, include_batch, device):
        z = torch.tensor(val, device=device)
        if len(z.shape) == 0:
            z = torch.tensor([val], device=device)
        if include_batch:
            z = z.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            z = z.unsqueeze(1).unsqueeze(2)
        return z

    #@abstractmethod
    def get_model(self, model_mode, device, config=None, load=False):
        model = self.get_model_(model_mode, device, config, load)
        if config.use_cuda:
            model = torch.nn.DataParallel(model)
        return model

    @abstractmethod
    def update_config(self, config):
        pass

    def get_train_transform(self):
        return self.get_train_transform_()

    def get_test_transform(self):
        return self.get_test_transform_()

    def get_without_transform(self):
        return self.get_without_transform_()

    def compute_transform_low_high(self):
        """
        Computes the transformed zero and one pixel values.

        Should typically not be used. Use get_transformed_zero_one() instead, which uses hard-coded values.
        Those hard-coded values are computed using this function in the script scripts/compute_dataset_stats.py
        """
        mean, std = self.get_mean_std()
        transform = transforms.Normalize(mean, std)
        channels = self.get_each_entry_shape()[0]
        low = torch.zeros(channels, 1, 1)
        high = low + 1
        print(torch.sum(low).item(), torch.sum(high).item())
        transformed_low = transform(low)#.item()
        transformed_high = transform(high)#.item()
        print(transformed_low, transformed_high)
        return transformed_low.squeeze().tolist(), transformed_high.squeeze().tolist()

    def get_mean_std(self):
        return self.get_mean_std_mixin()

    # Returns mean and std in shape [1, c, 1, 1] for easy tensor operations later
    def get_mean_std_correct_dims(self, include_batch):
        mean, std = self.get_mean_std()
        return self.get_correct_dims(mean, include_batch, self.device), self.get_correct_dims(std, include_batch,
                self.device)

    # Transforms epsilon = 1/256 using channel-specific transformation.
    # Pixels below this value are clipped to 0 in order to promote sparsity.
    # This makes sense for RGB or greyscale images, which have a resolution of only 1/256
    # Returns a tensor of shape [1, c, 1, 1]
    def get_batched_epsilon(self):
        mean, std = self.get_mean_std_correct_dims(include_batch=True)
        # epsilon = torch.ones_like(mean) / 256.
        # Set epsilon to 0.1 to make small pixel values go to 0
        epsilon = 0.1 * torch.ones_like(mean)
        print(epsilon, epsilon.shape)
        epsilon = (epsilon - mean) / std
        print(epsilon, epsilon.shape)
        return epsilon

    def get_optimizer_scheduler(self, config, model):
        raise NotImplementedError('Please implement this in a sublcass')

    # Takes batch of multi-channel images with pixels in the range [0, 1]
    # Performs mean-std transformation
    def transform_images(self, images):
        mean, std = self.get_mean_std_correct_dims(include_batch=True)
        return (images - mean) / std

    # Takes a batch of multi-channel images with pixels normalized by mean and std
    # Undoes that transform
    def undo_transform_images(self, images):
        mean, std = self.get_mean_std_correct_dims(include_batch=True)
        return mean + images * std

    # Get a batch of 100 images with 10 images per class
    def get_regular_batch(self, images, targets, num_classes, num_per_class):
        entries = []
        tgt_entries = []
        # shape = images.shape
        shape = self.get_each_entry_shape()
        image_zero = self.get_zero_correct_dims(include_batch=False)
        image_one = self.get_one_correct_dims(include_batch=False)
        for cls in range(num_classes):
            count = 0
            i = 0
            while count < num_per_class and i < targets.shape[0]:
                if targets[i].item() == cls:
                    entries.append(images[i])
                    tgt_entries.append(targets[i])
                    count += 1
                i += 1
            # Append cross X images if not enough entries for this class
            # All-zero images can be produced easily by our optimization algo,
            # But cross image is hard to be produced by accident
            for j in range(count, num_per_class):
                cross = get_cross(shape[2], shape[0], targets)
                entries.append(cross * image_one + image_zero)
                tgt_entries.append(torch.tensor(cls, device=targets.device))

        return torch.stack(entries), torch.stack(tgt_entries)
