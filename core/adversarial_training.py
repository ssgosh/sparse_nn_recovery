from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from utils.batched_tensor_view_data_loader import BatchedTensorViewDataLoader


class AdversarialTrainer:
    def __init__(self, real_data_train_loader, sparse_input_dataset_recoverer : SparseInputDatasetRecoverer,
                 model, opt_model):
        self.real_data_train_loader = real_data_train_loader
        self.sparse_input_dataset_recoverer = sparse_input_dataset_recoverer
        self.model = model
        self.opt_model = opt_model

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


    def generate_m_image_batches_train_one_epoch_adversarial(self, m):
        pass

    # Say adversarial image batch size is 32
    # and m is 10
    # Then 320 images will be generated
    # Say k is 100.
    # Then 100 batches of real data and 100 batches each of size 32 from above 320 images will be used for adversarial training
    def generate_m_image_batches_train_k_batches_adversarial(self, m, k):
        # First, generate m image batches
        self.sparse_input_dataset_recoverer.dataset_len = m * self.adv_training_batch_size
        images, targets = self.sparse_input_dataset_recoverer.recover_image_dataset()
        fake_class_targets = targets + self.sparse_input_dataset_recoverer.num_real_classes
        adversarial_train_loader = BatchedTensorViewDataLoader(self.adv_training_batch_size,
                                                               images, targets, fake_class_targets)
