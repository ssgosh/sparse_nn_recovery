
class AdversarialTrainer:
    def __init__(self, real_data_train_loader, sparse_input_dataset_recoverer, model, opt_model):
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


    def train_one_epoch_adversarial_and_regenerate_images(self):
        pass
