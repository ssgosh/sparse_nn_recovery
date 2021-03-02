import argparse

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from core.adversarial_dataset_manager import AdversarialDatasetManager
from core.mlabels import MLabels
from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from core.tblabels import TBLabels
from utils.batched_tensor_view_data_loader import BatchedTensorViewDataLoader
from utils.dataset_helper import DatasetHelper
from utils.infinite_dataloader import InfiniteDataLoader

import sys

from utils.metrics_helper import MetricsHelper

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
                self.trainer.epoch, self.trainer.next_real_batch * self.trainer.adv_training_batch_size,
                self.train_dataset_len * self.trainer.adv_training_batch_size,
                100. * self.trainer.next_real_batch / self.train_dataset_len, lossval)
            )
            sys.stdout.write('\r')
            if self.dry_run:
                sys.stdout.write('\n')
                sys.stdout.flush()
                raise ShouldBreak('Breaking because of dry run')


def get_soft_labels(batch_inputs):
    n = DatasetHelper.get_dataset().get_num_classes()
    N = batch_inputs.shape[0]
    adv_soft_labels = torch.empty(N, n, device=batch_inputs.device).fill_(1 / n)
    return adv_soft_labels


class AdversarialTrainer:
    valid_adversarial_classification_modes = ['fake-classes', 'max-entropy']

    @classmethod
    def add_command_line_arguments(cls, parser : argparse.ArgumentParser):
        # Add arguments specific to adversarial-epoch and adversarial-batches mode
        # Adversarial batch size is same as the train batch size
        parser.add_argument('--num-adversarial-train-batches', type=int, default=100,
                            metavar='k', help='Number of batches to train for before regenerating images if in '
                                              '"adversarial-batches" mode')
        parser.add_argument('--num-adversarial-images-batch-mode', type=int, default=1024,
                            metavar='m', help='Number of batches of images to generate in "adversarial-batches" mode')
        parser.add_argument('--num-adversarial-images-epoch-mode', type=int, default=10240,
                            metavar='m', help='Number of batches of images to generate in "adversarial-epoch" mode')
        parser.add_argument('--adversarial-classification-mode', type=str, default='max-entropy',
                            metavar='CMODE', choices=cls.valid_adversarial_classification_modes,
                            help='Whether to use fake classes or soft labels with equal probability on real classes '
                                 'for adversarial examples. One of: ' + ', '.join(cls.valid_adversarial_classification_modes))

    def __init__(self, real_data_train_loader, real_data_train_samples, sparse_input_dataset_recoverer: SparseInputDatasetRecoverer,
                 model, opt_model, adv_training_batch_size, device, log_interval, dry_run, early_epoch,
                 num_batches_early_epoch,
                 test_loader : DataLoader, lr_scheduler_model : StepLR,
                 adversarial_classification_mode : str):
        self.adv_training_batch_size = adv_training_batch_size  # Same batch size is used for both real and adversarial training
        self.real_data_train_loader = real_data_train_loader
        self.real_data_train_samples = real_data_train_samples
        self.sparse_input_dataset_recoverer = sparse_input_dataset_recoverer
        self.model = model
        self.opt_model = opt_model
        self.next_real_batch = 0
        self.epoch = 0
        self.device = device
        self.early_epoch = early_epoch
        self.num_batches_early_epoch = num_batches_early_epoch
        self.ckpt_saver = self.sparse_input_dataset_recoverer.ckpt_saver
        self.test_loader = test_loader
        self.valid_loader = self.test_loader    # For now, validation data is just test data. Will change later
        self.lr_scheduler_model = lr_scheduler_model
        self.adversarial_classification_mode = adversarial_classification_mode
        self.adv_use_soft_labels = adversarial_classification_mode == 'max-entropy'
        self.adv_criterion = F.kl_div if self.adv_use_soft_labels else F.nll_loss

        # Use a fixed iterator to iterate over the training dataset
        self.real_data_train_iterator = iter(self.real_data_train_loader)

        self.logger = TrainLogger(self, log_interval, dry_run)
        self.tbh = self.sparse_input_dataset_recoverer.tbh
        self.sparsity_mode = self.sparse_input_dataset_recoverer.sparsity_mode
        self.train_dataset_len = len(self.real_data_train_loader)
        self.metrics_helper : MetricsHelper = MetricsHelper.get(adversarial_classification_mode=adversarial_classification_mode)

        self.dataset_mgr = AdversarialDatasetManager(sparse_input_dataset_recoverer,
                                                     train_batch_size=adv_training_batch_size,
                                                     test_batch_size=test_loader.batch_size)

    # Train model on the given batch. Used for real data or adversarial data training
    def train_one_batch(self, batch_inputs, batch_targets):
        data, target = batch_inputs.to(self.device), batch_targets.to(self.device)
        self.opt_model.zero_grad()
        output = self.model(data)
        real_loss = F.nll_loss(output, target)
        loss = real_loss + self.model.get_weight_decay()
        loss.backward()
        self.opt_model.step()
        self.logger.log_batch(loss.item())
        avg_real_probs = self.metrics_helper.compute_reduce_prob(output, target)
        self.log_losses_to_tensorboard({'real_loss': real_loss.item(),
                                        'avg_real_probs' : avg_real_probs.item(),},
                                       TBLabels.PER_BATCH_ADV_AGGREGATE,
                                       self.global_step())

    def global_step(self):
        return self.epoch * self.get_batches_in_epoch() + self.next_real_batch

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
                                    adversarial_batch_inputs, adversarial_batch_targets,
                                    adv_soft_labels):
        # Move everything to gpu before performing tensor operations
        real_data = real_batch_inputs.to(self.device)
        adv_data = adversarial_batch_inputs.to(self.device)
        real_targets = real_batch_targets.to(self.device)
        adv_targets = adversarial_batch_targets.to(self.device)

        self.opt_model.zero_grad()

        real_output = self.model(real_data)
        adv_output = self.model(adv_data)

        real_loss = F.nll_loss(real_output, real_targets)
        adv_loss = F.kl_div(adv_output, adv_soft_labels) if self.adv_use_soft_labels else F.nll_loss(adv_output, adv_targets)
        loss = real_loss + adv_loss + self.model.get_weight_decay()

        loss.backward()
        self.opt_model.step()
        self.logger.log_batch(loss.item())

        # Compute Probabilities
        avg_real_probs = self.metrics_helper.compute_reduce_prob(real_output, real_targets)
        avg_adv_probs = self.metrics_helper.compute_reduce_prob(adv_output, adv_targets, adv_data=True)
        self.log_losses_to_tensorboard(
            {
            'real_loss': real_loss.item(),
            'adv_loss': adv_loss.item(),
            'avg_real_probs': avg_real_probs.item(),
            'avg_adv_probs': avg_adv_probs.item(),
            },
            TBLabels.PER_BATCH_ADV_AGGREGATE,
            self.global_step()
        )

        #batch_inputs = torch.cat([real_batch_inputs, adversarial_batch_inputs])
        #batch_targets = torch.cat([real_batch_targets, adversarial_batch_targets])
        #self.train_one_batch(batch_inputs, batch_targets)

    # One epoch of adversarial training
    # def train_one_epoch_adversarial(self):
    #    pass

    def generate_m_images_train_one_epoch_adversarial(self, m):
        adversarial_train_loader = self.dataset_mgr.get_new_train_loader(m)
        # Now train
        self.model.train()
        # Note that we're using the loader here directly and not the cached iterator.
        # This creates a new iterator for the for loop, and iterates over all elements from start to end
        count = 0
        for self.next_real_batch, (real_batch, adv_batch) in enumerate(zip(self.real_data_train_loader, adversarial_train_loader)):
            real_images, real_targets = real_batch
            fake_images, _, fake_targets = adv_batch
            # XXX: Set this to a tensor of shape (N, num_classes), with all values 1 / num_classes
            adv_soft_labels = get_soft_labels(fake_images)
            self.train_one_batch_adversarial(real_images, real_targets, fake_images, fake_targets, adv_soft_labels)
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
        adversarial_train_loader = self.dataset_mgr.get_new_train_loader(m)

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

    def log_losses_to_tensorboard(self, losses_dict, label, global_step):
        self.tbh.log_dict(label, losses_dict, global_step)

    def get_batches_in_epoch(self):
        if self.early_epoch:
            return self.num_batches_early_epoch
        else:
            return self.train_dataset_len

    # Testing code
    def test_real_data(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for count, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                if self.early_epoch and count >= 1:
                    #print(f'\nBreaking from test due to early epoch after {count} batches')
                    break

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def validate(self):
        """
        Validates performance on the following datasets:
            - real validation data (self.valid_loader)
            - all members of manager.train_sample
            - all members of manager.valid
        :return:
        """
        # Test on real validation data
        print()
        self.test_and_return_metrics(self.real_data_train_samples, data_type='real', acc=None).log(
            'real_train_samples',
            self.tbh,
            tb_agg_label = TBLabels.PER_EPOCH_ADV_AGGREGATE_TRAIN_OVERALL,
            tb_per_class_label = TBLabels.PER_EPOCH_ADV_PER_CLASS_TRAIN_OVERALL,
            global_step=self.epoch
        )

        self.test_and_return_metrics(self.valid_loader, data_type='real', acc=None).log(
            'real_validation_data',
            self.tbh,
            tb_agg_label = TBLabels.PER_EPOCH_ADV_AGGREGATE_VALIDATION_OVERALL,
            tb_per_class_label = TBLabels.PER_EPOCH_ADV_PER_CLASS_VALIDATION_OVERALL,
            global_step=self.epoch
        )

        # Test on past adversarial train data
        datasets = self.dataset_mgr.train_sample
        tb_agg_label = TBLabels.PER_EPOCH_ADV_AGGREGATE_TRAIN_OVERALL
        tb_agg_label_i = TBLabels.PER_EPOCH_ADV_AGGREGATE_TRAIN
        tb_per_class_label = TBLabels.PER_EPOCH_ADV_PER_CLASS_TRAIN_OVERALL
        tb_per_class_label_i = TBLabels.PER_EPOCH_ADV_PER_CLASS_TRAIN

        self.test_and_log_intermittent_datasets('adversarial_train_samples', datasets, tb_agg_label, tb_agg_label_i, tb_per_class_label,
                                                tb_per_class_label_i)

        # Test on past adversarial validation data
        self.test_and_log_intermittent_datasets('adversarial_validation_data', self.dataset_mgr.valid,
                                                TBLabels.PER_EPOCH_ADV_AGGREGATE_VALIDATION_OVERALL,
                                                TBLabels.PER_EPOCH_ADV_AGGREGATE_VALIDATION,
                                                TBLabels.PER_EPOCH_ADV_PER_CLASS_VALIDATION_OVERALL,
                                                TBLabels.PER_EPOCH_ADV_PER_CLASS_VALIDATION)
        print()

    def test_and_log_intermittent_datasets(self, prefix, datasets, tb_agg_label, tb_agg_label_i, tb_per_class_label,
                                           tb_per_class_label_i):
        overall_metrics = []
        for i, loader in enumerate(datasets):
            self.test_and_return_metrics(loader, data_type='adv', acc=overall_metrics).log(
                f"{prefix}_{i}",
                self.tbh,
                tb_agg_label=tb_agg_label_i(i),
                tb_per_class_label=tb_per_class_label_i(i),
                global_step=self.epoch
            )
        MetricsHelper.reduce(overall_metrics).log(
            f"{prefix}_overall",
            self.tbh,
            tb_agg_label=tb_agg_label,
            tb_per_class_label=tb_per_class_label,
            global_step=self.epoch
        )

    def test_and_return_metrics(self, loader, data_type, acc) -> MetricsHelper:
        """

        :param loader: DataLoader
        :param data_type: 'real', 'adv'
        :param tb_agg_label:
        :param tb_per_class_label:
        :param acc: dictionary in which to accumulate metrics. Useful for computing aggregate stats
        :return: dictionary containing the stats
        """
        self.model.eval()
        loss = 0
        correct = 0
        mlabels = MLabels(data_type)
        metrics = MetricsHelper.get(mlabels, self.adversarial_classification_mode)
        adv_data = data_type == 'adv'
        with torch.no_grad():
            for count, tup in enumerate(loader):

                if data_type == 'real': data, target = tup
                elif data_type == 'adv': data, _, target = tup # target is fake class here. _ is real class
                else: assert False, f"Invalid data_type {data_type}"

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                adv_soft_labels = get_soft_labels(data) if adv_data else None
                metrics.accumulate_batch_stats(output, target, adv_data=adv_data, adv_soft_labels=adv_soft_labels)
                #loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                #correct += pred.eq(target.view_as(pred)).sum().item()
                if self.early_epoch and count >= 1:
                    #print(f'\nBreaking from test due to early epoch after {count} batches')
                    break

        metrics.finalize_stats()
        if acc is not None: acc.append(metrics)
        #loss /= len(loader.dataset)
        #accuracy = correct / len(loader.dataset)

        return metrics

    def train_loop(self, num_epochs, train_mode, pretrain, config):
        assert train_mode in [ 'adversarial-epoch', 'adversarial-batches' ]
        for epoch in range(0, num_epochs):
            if pretrain and epoch == 0:
                print('Pre-training for 1 epoch')
                self.train_one_epoch_real()
                # train(args, model, device, train_loader, optimizer, epoch)
            else:
                if train_mode == 'adversarial-batches':
                    epoch_over = False
                    while not epoch_over:
                        epoch_over = self.generate_m_images_train_k_batches_adversarial(
                            m=config.num_adversarial_images_batch_mode, k=config.num_adversarial_train_batches)
                elif train_mode == 'adversarial-epoch':
                    self.generate_m_images_train_one_epoch_adversarial(
                        m=config.num_adversarial_images_epoch_mode)
            self.validate()
            self.ckpt_saver.save_model(self.model, epoch, config.model_classname)
            self.lr_scheduler_model.step()

    def log_stats(self, stats, epoch):
        self.tbh.log_dict('', stats, epoch)
