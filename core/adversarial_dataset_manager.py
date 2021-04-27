import torch
from torch.utils.data import DataLoader, TensorDataset

from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from utils.infinite_dataloader import InfiniteDataLoader
from utils.torchutils import safe_clone


def _combine(prev : DataLoader, new : DataLoader, beta : float) -> DataLoader:
    """
    Combines two datasets. Keeps only beta fraction of previous dataset.

    NOTE: As of now we do the combination in a single shot and not in batched mode
    :param prev: previous dataset. A dataloader.
    :param new: new dataset. A dataloader.
    """
    N = len(prev.dataset)
    train_batch_size = prev.batch_size
    prev_items = next(iter(DataLoader(prev.dataset, batch_size=N)))
    new_items = next(iter(   DataLoader(  new.dataset, batch_size=len(new.dataset)  )   ))
    assert(len(prev_items) == len(new_items))

    cutoff = int(beta * N)
    idx = torch.randperm(N)[0:cutoff]

    prev_items_sampled = []
    for item in prev_items:
        prev_items_sampled.append(item[idx])

    out_items = []
    for a, b in zip(prev_items_sampled, new_items):
        out_items.append(torch.cat([a, b]))

    return DataLoader(TensorDataset(*out_items), batch_size=train_batch_size)


class DatasetMerger:
    def __init__(self, beta, combine):
        self.beta = beta
        self.combine = combine
        self.last_combined_train = None
        self.last_generated_train : DataLoader = None

    def combine_with_previous_train(self, new_train) -> DataLoader:
        if not self.combine or self.last_combined_train is None:
            self.last_combined_train = self.last_generated_train = new_train
            return new_train
        self.last_combined_train = _combine(self.last_combined_train, new_train, self.beta)
        self.last_generated_train = new_train
        return self.last_combined_train

class AdversarialDatasetManager:
    def __init__(self,
                 sparse_input_dataset_recoverer : SparseInputDatasetRecoverer,
                 train_batch_size : int,
                 test_batch_size : int
                 ):
        self._dataset_epoch = 0
        self.dataset_count = 0
        self.sidr = sparse_input_dataset_recoverer
        # List of train, test and validation dataset loaders
        # XXX: Train dataset is not kept in a list so as to be gc'd by python
        #      Only current train dataset is kept.
        # Depending on the mode, current train data may be a geometric combination of previous train datasets
        self.train : DataLoader = None
        self.valid = []
        self.test = []

        # Also keep around just a sample from the training dataset
        self.train_sample = []

        # List of epoch numbers in which the corresponding dataset was generated
        self.dataset_generation_epochs = []

        # Merges new train data with previous
        self.dmerger = DatasetMerger(beta=0.7, combine=True)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size


    # Make sure that our code is only ever increasing the dataset epoch
    @property
    def dataset_epoch(self):
        return self._dataset_epoch

    @dataset_epoch.setter
    def dataset_epoch(self, val):
        assert val >= self._dataset_epoch
        self._dataset_epoch = val

    # Generates m images and returns a train loader from it
    # Returned train loader is infinite and does not pass on gradients to the images
    # One batch is also generated for validation
    # One more batch is generated for testing
    def generate_m_images(self, m, mode):
        print(f"Generating {mode} dataset #{self.dataset_epoch}, size = ", m)
        self.sidr.dataset_len = m  # * self.adv_training_batch_size
        images, targets, _probs = self.sidr.recover_image_dataset(mode, self.dataset_epoch)

        # Fake labels are real_label + num_classes. E.g. Fake digit 0 has class 10, fake digit 1 has class 11 and so on
        fake_class_targets = targets.detach() + self.sidr.num_real_classes

        # Infinite, batched data loader. We don't need to propagate gradients to the dataset here, hence not using
        # BatchedTensorViewDataLoader
        # adversarial_train_loader = BatchedTensorViewDataLoader(self.adv_training_batch_size,
        #                                                       images, targets, fake_class_targets)
        adversarial_dataset = torch.utils.data.TensorDataset(images, targets, fake_class_targets)
        if mode == 'train':
            if len(adversarial_dataset) != 0:
                loader = InfiniteDataLoader(adversarial_dataset, **{'batch_size': self.train_batch_size, 'shuffle': True})
                sample = self.get_sample(images, targets, fake_class_targets, self.sidr.batch_size, self.test_batch_size)
            else:
                print(f'Got empty dataset for {mode} in epoch {self.dataset_epoch}')
                loader, sample = None, None
            return loader, sample
        else:
            if len(adversarial_dataset) != 0:
                loader = DataLoader(adversarial_dataset, **{'batch_size': self.test_batch_size, 'shuffle': True})
            else:
                print(f'Got empty dataset for {mode} in epoch {self.dataset_epoch}')
                loader = None
            return loader

    def generate_train_test_validation_dataset(self, m, t=None, v=None):
        t = t if t is not None else self.sidr.batch_size
        v = v if v is not None else self.sidr.batch_size

        new_train, sample = self.generate_m_images(m, 'train')
        valid = self.generate_m_images(v, 'valid')
        test = self.generate_m_images(t, 'test')

        if new_train is None or sample is None or valid is None or test is None:
            print('*** Pruned all adversarial images in dataset generated in epoch', self.dataset_epoch, '***')
            return False

        self.train_sample.append(sample)
        self.valid.append(valid)
        self.test.append(test)

        # Merge new_train with previous train if enabled
        self.train = self.dmerger.combine_with_previous_train(new_train)
        print("Generated train :", len(new_train.dataset), "Combined train :", len(self.train.dataset))
        self.dataset_count += 1
        return True

    def get_sample(self, images, targets, fake_class_targets, sample_size, batch_size):
        sample_size = images.shape[0] if images.shape[0] < sample_size else sample_size
        batch_size = sample_size if sample_size < batch_size else batch_size
        print("Generating Sample, size =", sample_size, "batch size =", batch_size)
        with torch.no_grad():
            idx = torch.randperm(targets.shape[0])[0:sample_size]
            images = safe_clone(images[idx])
            targets = safe_clone(targets[idx])
            fake_class_targets = safe_clone(fake_class_targets[idx])
            return DataLoader(TensorDataset(images, targets, fake_class_targets),
                              **{'batch_size' : batch_size})

    def get_new_train_loader(self, m, generate_new):
        generated = False
        if generate_new:
            generated = self.generate_train_test_validation_dataset(m)
            if generated:
                self.dataset_generation_epochs.append(self.dataset_epoch)
            assert len(self.train_sample) == self.dataset_count
            assert len(self.valid) == self.dataset_count
            assert len(self.test) == self.dataset_count
        return self.train, generated

