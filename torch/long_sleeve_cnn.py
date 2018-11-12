import torch.nn as nn
import torch.nn.functional as F


# Defines CNN topology
class LongSleeveCnn(nn.Module):
    def __init__(self):
        super(LongSleeveCnn, self).__init__()
        self.predict = False

        # Input channels = 3 (128 * 128), output channels = 8 (128 * 128)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)

        # Input channels = 8 (128 * 128), output channels = 8 (128 * 128)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

        # Input channels = 8 (128 * 128), output channels = 8 (64 * 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Input channels = 8 (64 * 64), output channels = 8 (64 * 64)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

        # Input channels = 8 (64 * 64), output channels = 8 (32 * 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Input channels = 8 (32 * 32), output channels = 8 (32 * 32)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

        # Input channels = 8 (32 * 32), output channels = 8 (16 * 16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Input channels = 8 (16 * 16), output channels = 8 (16 * 16)
        self.conv5 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

        # Input channels = 8 (16 * 16), output channels = 8 (8 * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 512 (8 * 8 * 8) input features, 64 output features (see sizing flow below).
        self.fc1 = nn.Linear(8 * 8 * 8, 64)

        # Dropout layer(s): drops with probability 0.4
        self.dropout = nn.Dropout(p=0.4)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))
        x = self.pool4(x)
        # re-shape: from pool2 to fc1
        x = x.view(-1, 18 * 8 * 8)
        # fc1: from 1152 (18 * 8 * 8) to 64
        x = F.relu(self.fc1(x))
        # dropout: p = 0.5
        if not self.predict:
            x = self.dropout(x)
        # fc2: from 64 to 2
        return self.fc2(x)
