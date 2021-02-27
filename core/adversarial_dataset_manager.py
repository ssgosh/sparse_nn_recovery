import torch

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from utils.infinite_dataloader import InfiniteDataLoader


class AdversarialDatasetManager:
    def __init__(self, sparse_input_dataset_recoverer : SparseInputDatasetRecoverer):
        self.sidr = sparse_input_dataset_recoverer
        # List of train, test and validation datasets
        self.train = []
        self.valid = []
        self.test = []

    # Generates m images and returns a train loader from it
    # Returned train loader is infinite and does not pass on gradients to the images
    # One batch is also generated for validation
    # One more batch is generated for testing
    def generate_m_images(self, m, train_batch_size):
        self.sidr.dataset_len = m  # * self.adv_training_batch_size
        images, targets = self.sidr.recover_image_dataset()

        # Fake labels are real_label + num_classes. E.g. Fake digit 0 has class 10, fake digit 1 has class 11 and so on
        fake_class_targets = targets.detach() + self.sidr.num_real_classes

        # Infinite, batched data loader. We don't need to propagate gradients to the dataset here, hence not using
        # BatchedTensorViewDataLoader
        # adversarial_train_loader = BatchedTensorViewDataLoader(self.adv_training_batch_size,
        #                                                       images, targets, fake_class_targets)
        adversarial_dataset = torch.utils.data.TensorDataset(images,
                                                             targets, fake_class_targets)
        adversarial_train_loader = InfiniteDataLoader(adversarial_dataset,
                                                      **{'batch_size': train_batch_size, 'shuffle': True})
        return adversarial_train_loader

