from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from mnist_model import Net

def recover_image(model, num_steps):
    image = torch.randn(1, 1, 28, 28)
    image.requires_grad = True
    optimizer = optim.SGD([image], lr=0.1)

    # Target is the "0" digit
    target = torch.tensor([0])

    for i in range(1, num_steps+1):
        output = model(image)
        loss = F.nll_loss(output, target)
        optimizer.step()

model = Net()
model_state_dict = torch.load('mnist_cnn.pt')
model.load_state_dict(model_state_dict)
print(model)
summary(model, (1, 28, 28))

recover_image(model, 10)

