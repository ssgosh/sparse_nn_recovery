import torch
from icontract import require
from torch import optim
from torchvision import datasets, models
from torchvision.transforms import transforms
import torch.nn.functional as F

from datasets.dataset_helper import CIFARDatasetHelper
from pytorch_cifar.models import ResNet18
from utils import torchutils

# model with wrapped logsoftmax
class ModelLogSoftmax(torch.nn.Module):
    def __init__(_model):
        self._model = _model

    def forward(x):
        return F.log_softmax(self._model(x), dim=1)

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
        model = models.VGG16(pretrained=True)
        model.classifier[6].out_features = 10
        # freeze conv layers
        for param in model.features.parameters():
            param.requires_grad = False

        model = ModelLogSoftmax(model)

    def get_optimizer_scheduler(self, config, model):
        optimizer = optim.SGD(model._model.classifier.parameters(), lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLr(optimizer, lambda epoch : 1)
        return optimizer, scheduler

