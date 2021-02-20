import sys
sys.path.append(".")

from utils.metrics_helper import MetricsHelper
from utils.mnist_helper import compute_mnist_transform_low_high
from models.mnist_model import ExampleCNNNet

import torch
import json

sparsity = {}
images = torch.randn(10, 1, 28, 28)
targets = torch.arange(10)
model = ExampleCNNNet(num_classes=10)
output = model(images)
a, b = compute_mnist_transform_low_high()
mth = MetricsHelper(a, b)
mth.compute_sparsities(images, model , targets, sparsity)
print(json.dumps(sparsity, indent=2))
