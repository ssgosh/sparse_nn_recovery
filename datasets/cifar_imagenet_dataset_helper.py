import torch
from icontract import require
from torch import optim
from torchvision import datasets, models
from torchvision.transforms import transforms
import torch.nn.functional as F

from datasets.cifar_dataset_helper import CIFARDatasetHelper
from utils import torchutils

# model with wrapped logsoftmax
class ModelLogSoftmax(torch.nn.Module):
    def __init__(self, _model):
        super().__init__()
        self._model = _model

    def forward(self, x):
        return F.log_softmax(self._model(x), dim=1)

def stupid_function(epoch):
    return 1

class CIFARImageNetDatasetHelper(CIFARDatasetHelper):
    def __init__(self, name, subset, non_sparse):
        super().__init__(name, subset, non_sparse)

        self.usual_mean = [0.485, 0.456, 0.406]
        self.usual_std = [0.229, 0.224, 0.225]

        self.usual_zero = [-2.1179, -2.0357, -1.8044]
        self.usual_one = [2.2489, 2.4286, 2.6400]

    def get_model_(self, model_mode, device, config=None, load=False):
        assert model_mode == 'max-entropy'
        assert load == False
        model = models.vgg16(pretrained=config.use_imagenet_pretrained_model).to(device)
        #model.classifier[6].out_features = 10
        model.classifier[6] = torch.nn.Linear(4096, 10)

        if config.use_imagenet_pretrained_model:
            print('Using imagenet pretrained model')
            print('Freezing conv layers')
            # freeze conv layers
            for param in model.features.parameters():
                param.requires_grad = False

        model = ModelLogSoftmax(model).to(device)
        self.optimizer = optim.SGD(model._model.classifier.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, stupid_function)
        print(model)
        return model

    def get_optimizer_scheduler(self, config, model):
        return self.optimizer, self.scheduler

