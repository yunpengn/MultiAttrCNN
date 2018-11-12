import torch.nn as nn
import torch.nn.functional as F

numOfClasses = 20


# Defines CNN topology
class AgeCnn(nn.Module):
    def __init__(self):
        super(AgeCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, 5, padding=1, stride=1)
        self.fc1 = nn.Linear(16 * 30 * 30, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, numOfClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
