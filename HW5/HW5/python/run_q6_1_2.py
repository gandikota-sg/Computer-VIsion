import numpy as np
import scipy.io

from nn import *
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


#https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
#https://towardsdatascience.com/three-ways-to-build-a-neural-network-in-pytorch-8cea49f9a61a
#https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

# Load data from matrices
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

# Get Test, Train, and Valid data into x and y
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# Hyperparameters for Tuning
max_iters = 10
batch_size = 16
learning_rate = 0.002
hidden_size = 64

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

# Convert data into PyTorch tensors and DataLoader for batching
train_x = torch.tensor(train_x).float()
valid_x = torch.tensor(valid_x).float()

label = np.where(train_y == 1)[1]
valid_label = np.where(valid_y == 1)[1]

label = torch.tensor(label)
valid_label = torch.tensor(valid_label)

train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x, label),
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(valid_x, valid_label),
                                          batch_size=batch_size,
                                          shuffle=True)

# Find dimensions and classes for model
train_examples = train_x.shape[0]
valid_examples = valid_x.shape[0]
examples, dimension = train_x.shape
examples, classes = train_y.shape

#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class ConvNet(nn.Module):
    def __init__(self, num_classes=36):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(50 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 5 * 5 * 50)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Define the neural network model
model = ConvNet()

# Initialize arr and loss funmction
train_loss_arr, valid_loss_arr, train_acc_arr, valid_acc_arr = [[] for _ in range(4)]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    train_loss = 0
    acc = 0
    valid_loss = 0
    v_acc = 0

    # Training loop
    model.train()  # Set the model to training mode
    for train_idx, (x, label) in enumerate(train_loader):
        x = x.reshape(batch_size, 1, 32, 32)
        optimizer.zero_grad()
        res = model(x)
        loss = criterion(res, label)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(res, 1)
        acc += (pred == label).sum().item()
        train_loss += loss.item()
    train_acc = acc / train_examples

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad(): 
        for valid_idx, (x, label) in enumerate(test_loader):
            x = x.reshape(batch_size, 1, 32, 32)
            res = model(x)
            loss = criterion(res, label)
            _, pred = torch.max(res, 1)
            v_acc += (pred == label).sum().item()
            valid_loss += loss.item()
        valid_acc = v_acc / valid_examples

    # Append metrics to respective lists
    train_loss_arr.append(train_loss / (train_examples / batch_size))
    valid_loss_arr.append(valid_loss / (valid_examples / batch_size))
    train_acc_arr.append(train_acc)
    valid_acc_arr.append(valid_acc)

    if i % 2 == 0:
        print("Epoch: {:02d} \t Train Loss: {:.2f} \t Train Accuracy : {:.2f}".format(
            i, train_loss, train_acc))

# Plotting
plt.figure(0)
plt.plot(np.arange(max_iters), train_acc_arr, 'r', label = "Training Accuracy")
plt.plot(np.arange(max_iters), valid_acc_arr, 'b', label = "Validation Accuracy")
plt.legend()
plt.xlabel('Iterations')  
plt.ylabel('Accuracy') 
plt.ylim(0, 1)
plt.title('Training and Validation Accuracy') 
plt.show()

plt.figure(1)
plt.plot(np.arange(max_iters), train_loss_arr, 'r', label = "Training Loss")
plt.plot(np.arange(max_iters), valid_loss_arr, 'b', label = "Validation Loss")
plt.legend()
plt.ylabel('Loss') 
plt.xlabel('Iterations')  
plt.ylim(0, 4)
plt.title('Training and Validation Loss') 
plt.show()
