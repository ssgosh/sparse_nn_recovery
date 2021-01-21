import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.fc1_l1 = None 
        self.fc2_l1 = None

        # list of the above two tensors
        self.all_l1 = []

    def forward(self, x):
        # First flatten the input, which is [N, 1, 28, 28]
        # After flattening, it is [N, 784]
        x = self.flatten1(x)

        # First hidden layer
        x = self.fc1(x)
        x = self.relu1(x)

        # L1 norm computation
        self.fc1_l1 = torch.norm(x, 1) / torch.numel(x)
        self.all_l1.clear()
        self.all_l1.append(self.fc1_l1)

        x = self.dropout1(x)

        # Second hidden layer
        x = self.fc2(x)
        x = self.relu2(x)

        # Compute l1 here
        self.fc2_l1 = torch.norm(x, 1) / torch.numel(x)
        self.all_l1.append(self.fc2_l1)

        x = self.dropout2(x)

        # Output Layer
        x = self.fc3(x)
        output = self.logsoftmax(x)
        #output = F.log_softmax(x, dim=1)
        return output

class MLPNet3Layer(nn.Module):
    def __init__(self):
        super(MLPNet3Layer, self).__init__()
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.fc1_l1 = None 
        self.fc2_l1 = None
        self.fc3_l1 = None

        # list of the above two tensors
        self.all_l1 = []

    def forward(self, x):
        # First flatten the input, which is [N, 1, 28, 28]
        # After flattening, it is [N, 784]
        x = self.flatten1(x)

        # First hidden layer
        x = self.fc1(x)
        x = self.relu1(x)

        # L1 norm computation
        self.fc1_l1 = torch.norm(x, 1) / torch.numel(x)
        self.all_l1.clear()
        self.all_l1.append(self.fc1_l1)

        x = self.dropout1(x)

        # Second hidden layer
        x = self.fc2(x)
        x = self.relu2(x)

        # Compute l1 here
        self.fc2_l1 = torch.norm(x, 1) / torch.numel(x)
        self.all_l1.append(self.fc2_l1)

        x = self.dropout2(x)

        # Third hidden layer
        x = self.fc3(x)
        x = self.relu3(x)

        # Compute l1 here
        self.fc3_l1 = torch.norm(x, 1) / torch.numel(x)
        self.all_l1.append(self.fc3_l1)

        x = self.dropout3(x)

        # Output Layer
        x = self.fc4(x)
        output = self.logsoftmax(x)
        #output = F.log_softmax(x, dim=1)
        return output

if __name__ == '__main__':
    from torchsummary import summary
    model = MLPNet3Layer()
    print(model)
    summary(model, (1, 28, 28))

