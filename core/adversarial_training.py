import argparse

import torch
import torch.nn.functional as F

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from utils.batched_tensor_view_data_loader import BatchedTensorViewDataLoader
from utils.infinite_dataloader import InfiniteDataLoader

import sys


class ShouldBreak(Exception):
    pass


class TrainLogger:
    def __init__(self, trainer, log_interval, dry_run):
        self.trainer = trainer
        self.log_interval = log_interval
        self.train_dataset_len = len(trainer.real_data_train_loader)
        self.dry_run = dry_run

    def log_batch(self, lossval):
        if self.trainer.next_real_batch % self.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.trainer.epoch, self.trainer.next_real_batch * self.trainer.adv_training_batch_size, self.train_dataset_len,
                100. * self.trainer.next_real_batch / self.train_dataset_len, lossval)
            )
            sys.stdout.write('\r')
            if self.dry_run:
                sys.stdout.write('\n')
                sys.stdout.flush()
                raise ShouldBreak('Breaking because of dry run')


class AdversarialTrainer:
    @staticmethod
    def add_command_line_arguments(parser : argparse.ArgumentParser):
        # Add arguments specific to adversarial-epoch and adversarial-batches mode
        # Adversarial batch size is same as the train batch size
        parser.add_argument('--num-adversarial-train-batches', type=int, default=100,
                            metavar='k', help='Number of batches to train for before regenerating images if in '
                                              '"adversarial-batches" mode')
        parser.add_argument('--num-adversarial-images-batch-mode', type=int, default=1024,
                            metavar='m', help='Number of batches of images to generate in "adversarial-batches" mode')
        parser.add_argument('--num-adversarial-images-epoch-mode', type=int, default=10240,
                            metavar='m', help='Number of batches of images to generate in "adversarial-epoch" mode')

    def __init__(self, real_data_train_loader, sparse_input_dataset_recoverer: SparseInputDatasetRecoverer,
                 model, opt_model, adv_training_batch_size, device, log_interval, dry_run, early_epoch,
                 num_batches_early_epoch):
        self.adv_training_batch_size = adv_training_batch_size  # Same batch size is used for both real and adversarial training
        self.real_data_train_loader = real_data_train_loader
        self.sparse_input_dataset_recoverer = sparse_input_dataset_recoverer
        self.model = model
        self.opt_model = opt_model
        self.next_real_batch = 0
        self.epoch = 0
        self.device = device
        self.early_epoch = early_epoch
        self.num_batches_early_epoch = num_batches_early_epoch
        self.ckpt_saver = self.sparse_input_dataset_recoverer.ckpt_saver

        # Use a fixed iterator to iterate over the training dataset
        self.real_data_train_iterator = iter(self.real_data_train_loader)

        self.logger = TrainLogger(self, log_interval, dry_run)
        self.tbh = self.sparse_input_dataset_recoverer.tbh
        self.sparsity_mode = self.sparse_input_dataset_recoverer.sparsity_mode

    # Train model on the given batch. Used for real data or adversarial data training
    def train_one_batch(self, batch_inputs, batch_targets):
        data, target = batch_inputs.to(self.device), batch_targets.to(self.device)
        self.opt_model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target) + self.model.get_weight_decay()
        loss.backward()
        self.opt_model.step()
        self.logger.log_batch(loss.item())

    # Train model on only real data for one full epoch. Used for pre-training.
    def train_one_epoch_real(self):
        count = 0
        for self.next_real_batch, real_batch in enumerate(self.real_data_train_loader):
            real_images, real_targets = real_batch
            self.train_one_batch(real_images, real_targets)
            count += 1
            if self.early_epoch and count >= self.num_batches_early_epoch:
                print(f'\nBreaking due to early epoch after {count} batches')
                self.next_real_batch = 0  # Need to reset this, else next epoch will immediately early-break
                break
        self.epoch += 1

    # Create a batch from real and adversarial data and call train_one_batch
    def train_one_batch_adversarial(self, real_batch_inputs, real_batch_targets,
                                    adversarial_batch_inputs, adversarial_batch_targets):
        # Move everything to gpu before performing tensor operations
        real_data = real_batch_inputs.to(self.device)
        adv_data = adversarial_batch_inputs.to(self.device)
        real_targets = real_batch_targets.to(self.device)
        adv_targets = adversarial_batch_targets.to(self.device)

        self.opt_model.zero_grad()

        real_output = self.model(real_data)
        adv_output = self.model(adv_data)

        real_loss = F.nll_loss(real_output, real_targets)
        adv_loss = F.nll_loss(adv_output, adv_targets)
        loss = real_loss + adv_loss + self.model.get_weight_decay()

        loss.backward()
        self.opt_model.step()
        self.logger.log_batch(loss.item())
        self.log_losses_to_tensorboard({'real_loss': real_loss.item(), 'adv_loss': adv_loss.item()}, self.next_real_batch)

        #batch_inputs = torch.cat([real_batch_inputs, adversarial_batch_inputs])
        #batch_targets = torch.cat([real_batch_targets, adversarial_batch_targets])
        #self.train_one_batch(batch_inputs, batch_targets)

    # One epoch of adversarial training
    # def train_one_epoch_adversarial(self):
    #    pass

    # Generates m images and returns a train loader from it
    # Returned train loader is infinite and does not pass on gradients to the images
    def generate_m_images(self, m):
        self.sparse_input_dataset_recoverer.dataset_len = m #* self.adv_training_batch_size
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
                                                      **{'batch_size': self.adv_training_batch_size, 'shuffle': True})
        return adversarial_train_loader

    def generate_m_images_train_one_epoch_adversarial(self, m):
        adversarial_train_loader = self.generate_m_images(m)
        # Now train
        self.model.train()
        # Note that we're using the loader here directly and not the cached iterator.
        # This creates a new iterator for the for loop, and iterates over all elements from start to end
        count = 0
        for self.next_real_batch, (real_batch, adv_batch) in enumerate(zip(self.real_data_train_loader, adversarial_train_loader)):
            real_images, real_targets = real_batch
            fake_images, _, fake_targets = adv_batch
            self.train_one_batch_adversarial(real_images, real_targets, fake_images, fake_targets)
            count += 1
            if self.early_epoch and count >= self.num_batches_early_epoch:
                print(f'\nBreaking due to early epoch after {count} batches')
                break

        self.epoch += 1

    # Say adversarial image batch size is 32
    # and m is 10
    # Then 320 images will be generated
    # Say k is 100.
    # Then 100 batches of real data and 100 batches each of size 32 from above 320 images
    # will be used for adversarial training
    def generate_m_images_train_k_batches_adversarial(self, m, k):
        adversarial_train_loader = self.generate_m_images(m)

        # Need to set this before training
        self.model.train()
        # stopped_early keeps track of whether m batches of real data were completed or not before
        # It may happen that we run through the end of the real dataset before we finish m batches.
        # In that case, we'll get another iterator to the loader
        stopped_early = True
        epoch_over = False
        early_epoch = False
        i = 0
        while stopped_early:
            for (real_batch, adv_batch) in zip(self.real_data_train_iterator, adversarial_train_loader):
                real_images, real_targets = real_batch
                fake_images, _, fake_targets = adv_batch
                self.train_one_batch_adversarial(real_images, real_targets, fake_images, fake_targets)
                i += 1
                self.next_real_batch += 1  # Keep track of batch number
                # Say total batches = 1000
                # early epoch batches = 10
                # k = 100
                # This will make sure we break at 10th batch if early epoch is set
                early_epoch = self.early_epoch and self.next_real_batch >= self.num_batches_early_epoch
                if early_epoch:
                    print(f'\nBreaking due to early epoch after {self.next_real_batch} batches')
                    break
                if i >= k: # This breaks at 100th batch,
                    stopped_early = False
                    sys.stdout.write('\n')  # For log_batch stuff to persist on console
                    break
            if stopped_early or early_epoch: # Rolls over to the next epoch
                # Get another iterator to the dataset since this one is done
                # This will shuffle the dataset automatically if it's set in the train loader kwargs
                self.real_data_train_iterator = iter(self.real_data_train_loader)
                self.epoch += 1
                epoch_over = True
                self.next_real_batch = 0
            if early_epoch:
                break

        return epoch_over

    def log_losses_to_tensorboard(self, losses_dict, global_step):
        self.tbh.log_dict(f"{self.sparsity_mode}", losses_dict, global_step)
