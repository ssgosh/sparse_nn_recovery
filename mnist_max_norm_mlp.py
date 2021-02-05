import torch
import torch.nn as nn
import torch.nn.functional as F

from max_norm_layer import MaxNormLayer

class MaxNormMLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MaxNormMLP, self).__init__()
        self.flatten1 = nn.Flatten()

        self.mn1 = MaxNormLayer(784, 256)
        #self.fc1 = nn.Linear(256, 256)

        #self.mn2 = MaxNormLayer(256, 128)
        #self.fc2 = nn.Linear(128, 128)

        #self.mn3 = MaxNormLayer(128, 128)

        self.output = nn.Linear(256, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Needed for adv training etc
        self.all_l1 = []

    def forward(self, x):
        # First flatten the input, which is [N, 1, 28, 28]
        # After flattening, it is [N, 784]
        x = self.flatten1(x)

        # First hidden layer
        x = self.mn1(x)
        #x = self.fc1(x)

        # Second hidden layer
        #x = self.mn2(x)
        #x = self.fc2(x)

        # Third hidden layer
        #x = self.mn3(x)

        # Output Layer
        x = self.output(x)
        output = self.logsoftmax(x)
        return output

    def get_weight_decay(self):
        return self.mn1.get_weight_decay()
