import torch
from torch.utils.data import DataLoader, TensorDataset

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from utils.infinite_dataloader import InfiniteDataLoader


def safe_clone(x):
    return torch.clone(x.detach())


class AdversarialDatasetManager:
    def __init__(self,
                 sparse_input_dataset_recoverer : SparseInputDatasetRecoverer,
                 train_batch_size : int,
                 test_batch_size : int
                 ):
        self.dataset_epoch = 0
        self.sidr = sparse_input_dataset_recoverer
        # List of train, test and validation dataset loaders
        # XXX: Train dataset is not kept in a list so as to be gc'd by python
        #      Only current train dataset is kept.
        self.train = None
        self.valid = []
        self.test = []

        # Also keep around just a sample from the training dataset
        self.train_sample = []

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    # Generates m images and returns a train loader from it
    # Returned train loader is infinite and does not pass on gradients to the images
    # One batch is also generated for validation
    # One more batch is generated for testing
    def generate_m_images(self, m, mode):
        print(f"Generating {mode} dataset #{self.dataset_epoch}, size = ", m)
        self.sidr.dataset_len = m  # * self.adv_training_batch_size
        images, targets = self.sidr.recover_image_dataset(mode, self.dataset_epoch)

        # Fake labels are real_label + num_classes. E.g. Fake digit 0 has class 10, fake digit 1 has class 11 and so on
        fake_class_targets = targets.detach() + self.sidr.num_real_classes

        # Infinite, batched data loader. We don't need to propagate gradients to the dataset here, hence not using
        # BatchedTensorViewDataLoader
        # adversarial_train_loader = BatchedTensorViewDataLoader(self.adv_training_batch_size,
        #                                                       images, targets, fake_class_targets)
        adversarial_dataset = torch.utils.data.TensorDataset(images, targets, fake_class_targets)
        if mode == 'train':
            loader = InfiniteDataLoader(adversarial_dataset, **{'batch_size': self.train_batch_size, 'shuffle': True})
            sample = self.get_sample(images, targets, fake_class_targets, self.sidr.batch_size, self.test_batch_size)
            return loader, sample
        else:
            loader = DataLoader(adversarial_dataset, **{'batch_size': self.test_batch_size, 'shuffle': True})
            return loader

    def generate_train_test_validation_dataset(self, m, t=None, v=None):
        t = t if t is not None else self.sidr.batch_size
        v = v if v is not None else self.sidr.batch_size

        self.train, sample = self.generate_m_images(m, 'train')
        self.train_sample.append(sample)
        self.valid.append(self.generate_m_images(v, 'valid'))
        self.test.append(self.generate_m_images(t, 'test'))

    def get_sample(self, images, targets, fake_class_targets, sample_size, batch_size):
        batch_size = sample_size if sample_size < batch_size else batch_size
        print("Generating Sample, size =", sample_size, "batch size =", batch_size)
        with torch.no_grad():
            idx = torch.randperm(targets.shape[0])[0:sample_size]
            images = safe_clone(images[idx])
            targets = safe_clone(targets[idx])
            fake_class_targets = safe_clone(fake_class_targets[idx])
            return DataLoader(TensorDataset(images, targets, fake_class_targets),
                              **{'batch_size' : batch_size})

    def get_new_train_loader(self, m):
        self.generate_train_test_validation_dataset(m)
        self.dataset_epoch += 1
        assert len(self.train_sample) == self.dataset_epoch
        assert len(self.valid) == self.dataset_epoch
        assert len(self.test) == self.dataset_epoch
        return self.train

