import torch.nn as nn
import torch.nn.functional as F


# Defines CNN topology
class GenderCnn(nn.Module):
    def __init__(self):
        super(GenderCnn, self).__init__()

        # Input channels = 3 (32 * 32), output channels = 18 (32 * 32)
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        # Input channels = 18 (32 * 32), output channels = 18 (16 * 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Input channels = 18 (16 * 16), output channels = 18 (16 * 16)
        self.conv2 = nn.Conv2d(18, 18, kernel_size=5, stride=1, padding=2)

        # Input channels = 18 (16 * 16), output channels = 18 (8 * 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 1152 (18 * 8 * 8) input features, 64 output features (see sizing flow below).
        self.fc1 = nn.Linear(18 * 8 * 8, 64)

        # Dropout layer: drops with probability 0.4
        self.dropout = nn.Dropout(p=0.4)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # conv1: from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        # pool1: from (18, 32, 32) to (18, 16, 16)
        x = self.pool1(x)
        # conv2: from (18, 16, 16) to (18, 16, 16)
        x = F.relu(self.conv2(x))
        # pool2: from (18, 16, 16) to (18, 8, 8)
        x = self.pool2(x)
        # re-shape: from pool2 to fc1
        x = x.view(-1, 18 * 8 * 8)
        # fc1: from 1152 (18 * 8 * 8) to 64
        x = F.relu(self.fc1(x))
        # dropout: p = 0.4
        x = self.dropout(x)
        # fc2: from 64 to 2
        return self.fc2(x)
