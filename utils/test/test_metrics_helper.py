import sys
sys.path.append(".")

import utils.metrics_helper as mth
from models.mnist_model import ExampleCNNNet

import torch
import json

sparsity = {}
images = torch.randn(10, 1, 28, 28)
targets = torch.arange(10)
model = ExampleCNNNet(num_classes=10)
output = model(images)
mth.compute_sparsities(images, model , targets, sparsity)
print(json.dumps(sparsity, indent=2))
