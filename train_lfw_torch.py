import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Defines CNN topology
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        # Input channels = 3 (32 * 32), output channels = 18 (32 * 32)
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        # Input channels = 18 (32 * 32), output channels = 18 (16 * 16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Input channels = 18 (16 * 16), output channels = 18 (16 * 16)
        self.conv2 = torch.nn.Conv2d(18, 18, kernel_size=5, stride=1, padding=2)

        # Input channels = 18 (16 * 16), output channels = 18 (8 * 8)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 1152 (18 * 8 * 8) input features, 64 output features (see sizing flow below).
        self.fc1 = torch.nn.Linear(18 * 8 * 8, 64)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 2)

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
        # fc2: from 64 to 2
        return self.fc2(x)


def train_net(net, train_loader, val_loader, n_epochs, loss_fn, optimizer, batch_size=32, learning_rate=0.001, print_every=2000):
    for epoch in range(n_epochs):
        # Uses this variable to take note of the loss in the current round.
        running_loss = 0.0

        # Iterates through every data point in the training data-set.
        for i, data in enumerate(train_loader, 0):
            # Get inputs.
            inputs, labels = data
            # Set the parameter gradients to zero.
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize.
            outputs = net(inputs)
            loss_size = loss_fn(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Updates the loss value.
            running_loss += loss_fn.item()

            # Prints the average loss of every 2000 mini-batches.
            if i % print_every == (print_every - 1):
                print("epoch=%d i=%d loss=%.3f" % epoch, i, running_loss / print_every)
                running_loss = 0.0
        print("Finished the training of %d epoch(es)." % epoch)

        # Iterates through every data point in the validation data-set.
        running_loss = 0.0
        for inputs, labels in val_loader:
            val_outputs = net(inputs)
            val_loss_size = loss_fn(val_outputs, labels)
            running_loss += val_loss_size.item()
        print("epoch=%d validation loss=%.3f" % (epoch, running_loss / len(val_loader)))

    print("Finished the training of all epoch(es).")


# Initializes a CNN instance.
net = ConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
