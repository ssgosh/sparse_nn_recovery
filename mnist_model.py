import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # Added for computing layer-wise l1 penalty
        # Will be computed during forward pass
        # Taking l1 of max pool layers and not conv or relu layers
        # This enforces a stronger condition that sparsity should occur
        # together in 2x2 blocks
        #
        # Value of 0 in conv doesn't tell us much - we are concerned more
        # about whether the non-linearity will be activated or not.
        #
        # Enforcing sparsity at relu layer makes sense - though it won't
        # enforce collective sparsity like enforcing it on the max-pool layer
        # would.
        self.max_pool1_l1 = None
        self.max_pool2_l1 = None
        self.fc1_l1 = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # TODO: compute l1 here
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # TODO: compute l1 here
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # TODO: compute l1 here
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

