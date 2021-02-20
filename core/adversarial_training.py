import torch

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from utils.batched_tensor_view_data_loader import BatchedTensorViewDataLoader
from utils.infinite_dataloader import InfiniteDataLoader


class AdversarialTrainer:
    def __init__(self, real_data_train_loader, sparse_input_dataset_recoverer : SparseInputDatasetRecoverer,
                 model, opt_model, adv_training_batch_size):
        self.adv_training_batch_size = adv_training_batch_size
        self.real_data_train_loader = real_data_train_loader
        self.sparse_input_dataset_recoverer = sparse_input_dataset_recoverer
        self.model = model
        self.opt_model = opt_model
        self.next_real_batch = 0

        # Use a fixed iterator to iterate over the training dataset
        self.real_data_train_iterator = iter(self.real_data_train_loader)

    # Train model on the given batch. Used for real data or adversarial data training
    def train_one_batch(self, batch_inputs, batch_targets):
        pass

    # Train model on only real data for one full epoch. Used for pre-training.
    def train_one_epoch_real(self):
        pass

    # Create a batch from real and adversarial data and train
    def train_one_batch_adversarial(self, real_batch_inputs, real_batch_targets,
                                    adversarial_batch_inputs, adversarial_batch_targets):
        pass

    # One epoch of adversarial training
    def train_one_epoch_adversarial(self):
        pass

    # Generates m image batches and returns a train loader from it
    # Say adversarial image batch size is 32
    # and m is 10
    # Then 320 images will be generated
    # Returned train loader is infinite and does not pass on gradients to the images
    def generate_m_image_batches(self, m):
        self.sparse_input_dataset_recoverer.dataset_len = m * self.adv_training_batch_size
        images, targets = self.sparse_input_dataset_recoverer.recover_image_dataset()

        # Fake labels are real_label + num_classes. E.g. Fake digit 0 has class 10, fake digit 1 has class 11 and so on
        fake_class_targets = targets.detach() + self.sparse_input_dataset_recoverer.num_real_classes

        # Infinite, batched data loader. We don't need to propagate gradients to the dataset here, hence not using
        # BatchedTensorViewDataLoader
        # adversarial_train_loader = BatchedTensorViewDataLoader(self.adv_training_batch_size,
        #                                                       images, targets, fake_class_targets)
        adversarial_dataset = torch.utils.data.TensorDataset(images,
                                                             targets, fake_class_targets)
        adversarial_train_loader = InfiniteDataLoader(adversarial_dataset,
                                                      **{'batch_size': self.adv_training_batch_size})
        return adversarial_train_loader


    def generate_m_image_batches_train_one_epoch_adversarial(self, m):
        pass

    # Say adversarial image batch size is 32
    # and m is 10
    # Then 320 images will be generated
    # Say k is 100.
    # Then 100 batches of real data and 100 batches each of size 32 from above 320 images will be used for adversarial training
    def generate_m_image_batches_train_k_batches_adversarial(self, m, k):
        adversarial_train_loader = self.generate_m_image_batches(m)
        # stopped_early keeps track of whether m batches of real data were completed or not before
        # It may happen that we run through the end of the real dataset before we finish m batches.
        # In that case, we'll get another iterator to the loader
        stopped_early = True
        while stopped_early:
            for i, (real_batch, adv_batch) in enumerate(zip(self.real_data_train_iterator, adversarial_train_loader)):
                real_images, real_targets = real_batch
                fake_images, _, fake_targets = adv_batch
                self.train_one_batch_adversarial(real_images, real_targets, fake_images, fake_targets)
                self.next_real_batch += 1 # Keep track of batch number
                if i >= m:
                    stopped_early = False
                    break
            if stopped_early:
                # Get another iterator to the dataset since this one is done
                self.real_data_train_iterator = iter(self.real_data_train_loader)
