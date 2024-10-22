import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torchvision.transforms import transforms

from datasets.dataset_helper import DatasetHelper
from datasets.non_sparse_normalization_mixin import NonSparseNormalizationMixin
from models.mnist_model import ExampleCNNNet
from utils import mnist_helper
from utils import torchutils


class MNISTdatasetHelper(DatasetHelper, ):
    def __init__(self, name, subset, non_sparse):
        super().__init__(name, subset, non_sparse)

        # These implement the non-sparse normalization mixin
        self.train_transforms = [transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),]
        self.test_transforms = []
        self.usual_mean = (0.1307, )
        self.usual_std = (0.3081, )
        self.non_sparse_mean = { 0.3 : (0.4032246160182823, ) }
        self.non_sparse_std =  { 0.3 : (0.23552181428064123, ) }
        self.constant_pixel_val = 0.3

        # Further needed by the non-sparse mixin
        # These are transformed values of zero and one pixel values
        self.usual_zero = -0.4242129623889923
        self.usual_one = 2.821486711502075

        self.non_sparse_zero = { 0.3 : -1.7120479345321655 }
        self.non_sparse_one =  { 0.3 : 2.533843517303467 }

    def get_dataset_(self, path, which, transform):
        return datasets.MNIST(path, train=(which == 'train'),
                transform=transform, download=True)


    def get_num_classes(self):
        return 10

    def get_each_entry_shape(self):
        return (1, 28, 28)

    def get_model_(self, model_mode, device, config=None, load=False):
        if model_mode == 'fake-classes': model = ExampleCNNNet(20).to(device)
        elif model_mode == 'max-entropy': model = ExampleCNNNet(10).to(device)
        else: raise ValueError(f"Invalid model mode {model_mode}")
        if load:
            # Load model from state dictionary
            checkpoint = torch.load(config.discriminator_model_file, map_location=torch.device(device))
            if 'model' in checkpoint:
                checkpoint = torchutils.load_data_parallel_state_dict_as_normal(checkpoint['model'])
            model.load_state_dict(checkpoint)
        # model = MLPNet().to(device)
        # model = MLPNet3Layer(num_classes=20).to(device)
        # model = MaxNormMLP(num_classes=20).to(device)
        return model

    def update_config(self, config):
        config.model_classname = 'ExampleCNNNet'
        config.mnist_lr_step_size = 1
        if not hasattr(config, 'discriminator_model_file') or config.discriminator_model_file is None:
            config.discriminator_model_file = 'ckpt/mnist_cnn.pt'

    def get_optimizer_scheduler(self, config, model):
        optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
        # Learning rate scheduler
        scheduler = StepLR(optimizer, step_size=config.mnist_lr_step_size, gamma=config.gamma)
        return optimizer, scheduler

