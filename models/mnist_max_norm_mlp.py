import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.max_norm_layer import MaxNormLayer

class MaxNormMLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MaxNormMLP, self).__init__()
        self.flatten1 = nn.Flatten()

        self.mn1 = MaxNormLayer(784, 512, lambd=0.01)
        self.dropout1 = nn.Dropout(0.25)

        #self.fc1 = nn.Linear(512, 256)
        #self.relu1 = nn.ReLU()
        #self.dropout1 = nn.Dropout(0.5)

        #self.fc2 = nn.Linear(256, 128)
        #self.relu2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        self.mn2 = MaxNormLayer(512, 256, lambd=0.001)
        self.dropout2 = nn.Dropout(0.5)

        #self.fc3 = nn.Linear(256, 256)

        #self.mn3 = MaxNormLayer(256, 128)

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
        x = self.dropout1(x)

        #x = self.fc1(x)
        #x = self.relu1(x)
        #x = self.dropout1(x)

        #x = self.fc2(x)
        #x = self.relu2(x)
        #x = self.dropout2(x)

        # Second hidden layer
        x = self.fc2(x)
        x = self.mn2(x)
        x = self.dropout2(x)
        #x = self.fc2(x)

        # Third hidden layer
        #x = self.mn3(x)

        # Output Layer
        x = self.output(x)
        output = self.logsoftmax(x)
        return output

    def get_weight_decay(self):
        return self.mn1.get_weight_decay()

if __name__ == '__main__':
    model = MaxNormMLP()
    x = torch.randn(10, 1, 28, 28)
    y = model(x)

