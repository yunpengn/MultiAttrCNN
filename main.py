from convolution_net import ConvolutionNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def train_net(net, train_loader, val_loader, n_epochs, loss_fn, optimizer, print_every=2000):
    for epoch in range(n_epochs):
        total_train_loss = 0.0

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
            total_train_loss += loss_size.item()

        print("Finished the training of %d epoch(es)." % epoch)
        storage_path = 'LFW_model_torch/cnn_epoch{}.pkl'.format(epoch)
        torch.save(net.state_dict(), storage_path)
        print("Saved the network at %s." % storage_path)

        # Iterates through every data point in the validation data-set.
        total_val_loss = 0.0
        total_count = 0
        correct_count = 0
        for inputs, label in val_loader:
            val_outputs = net(inputs)
            _, predicted = torch.max(val_outputs.data, 1)
            total_count += label.size(0)
            correct_count += (predicted == label).sum()

            val_loss_size = loss_fn(val_outputs, label)
            total_val_loss += val_loss_size.item()

        print("epoch=%d training loss=%.3f." % (epoch, total_train_loss / len(train_loader)))
        print("epoch=%d validation loss=%.3f." % (epoch, total_val_loss / len(val_loader)))
        print("epoch=%d accuracy=%.3f." % (epoch, correct_count / total_count))

    print("Finished the training of all epoch(es).")


# Creates the data-set for training and validation.
transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_set = ImageFolder("LFW_extract/train", transformer)
print("Finished loading the training data-set.")
val_set = ImageFolder("LFW_extract/val", transformer)
print("Finished loading the validation data-set.")

# Creates the data-loader for training and validation.
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
print("Finished creating the training data-loader.")
val_loader = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=2)
print("Finished creating the validation data-loader.")

# Initializes a CNN instance.
cnn_net = ConvolutionNet()
# Defines the loss function.
loss_fn = nn.CrossEntropyLoss()
# Uses a SGD-based optimizer.
sgd_optimizer = optim.SGD(cnn_net.parameters(), lr=0.001, momentum=0.9)

# Starts the training and validation of the model.
print("Going to train the model...")
train_net(cnn_net, train_loader, val_loader, 2, loss_fn, sgd_optimizer)
print("Finished training the model...")
