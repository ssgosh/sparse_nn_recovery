import torch
from icontract import require
from torch import optim
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F

from datasets.dataset_helper import DatasetHelper
from pytorch_cifar.models import ResNet18
from utils import torchutils


class CIFARDatasetHelper(DatasetHelper):
    @require(lambda non_sparse: not non_sparse)
    def __init__(self, name, subset, non_sparse):
        super().__init__(name, subset, non_sparse)

        # These implement the non-sparse normalization mixin
        self.train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # XXX: RandomAffine is not used here: https://github.com/kuangliu/pytorch-cifar/issues/130
            #transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),
        ]
        self.test_transforms = []
        self.usual_mean = (0.4914, 0.4822, 0.4465)
        # XXX: This is what is used in https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py#L34
        #   However, it does not match the computed std deviation values.
        #   Computed values are:
        #   mean = [0.4913996458053589, 0.48215845227241516, 0.44653093814849854]
        #   std = [0.2470322549343109, 0.24348513782024384, 0.26158788800239563]
        self.usual_std = (0.2023, 0.1994, 0.2010)
        self.non_sparse_mean = {}
        self.non_sparse_std = {}
        self.constant_pixel_val = 0.

        # Further needed by the non-sparse mixin
        # These are transformed values of zero and one pixel values
        self.usual_zero = [-2.429065704345703, -2.418254852294922, -2.22139310836792]
        self.usual_one = [2.5140879154205322, 2.596790313720703, 2.7537312507629395]

        self.non_sparse_zero = {}
        self.non_sparse_one = {}

    def get_dataset_(self, path, which, transform):
        return datasets.CIFAR10(path, download=True, train=(which == 'train'), transform=transform)

    def get_num_classes(self):
        return 10

    def get_each_entry_shape(self):
        return (3, 32, 32)

    def get_model(self, model_mode, device, config=None, load=False):
        model = None
        if model_mode == 'fake-classes':
            raise ValueError(f"Model mode {model_mode} not supported for CIFAR10")
        elif model_mode == 'max-entropy':
            assert load
            print('Loading model from', config.discriminator_model_file)
            model = ResNet18().to(device)
            checkpoint = torch.load(config.discriminator_model_file, map_location=torch.device(device))
            model_state_dict = torchutils.load_data_parallel_state_dict_as_normal(checkpoint['net'])
            model.load_state_dict(model_state_dict)
        else:
            raise ValueError(f"Invalid model mode {model_mode}")
        # model = MLPNet().to(device)
        # model = MLPNet3Layer(num_classes=20).to(device)
        # model = MaxNormMLP(num_classes=20).to(device)
        return model

    def update_config(self, config):
        config.model_classname = 'ResNet18'
        config.discriminator_model_file = 'ckpt/ResNet18/ckpt.pth'
        config.cifar_lr = 0.1
        config.cifar_momentum = 0.9
        config.cifar_weight_decay = 5e-4
        config.cifar_lr_scheduler = 'CosineAnnealingLR'
        config.cifar_t_max = 200

    def get_optimizer_scheduler(self, config, model):
        optimizer = optim.SGD(model.parameters(), lr=config.cifar_lr,
                              momentum=config.cifar_momentum, weight_decay=config.cifar_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cifar_t_max)
        return optimizer, scheduler

