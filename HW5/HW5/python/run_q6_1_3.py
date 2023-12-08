import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# Transformations for image normalization
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Hyperparameters for Tuning
max_iters = 20
batch_size = 16
learning_rate = 0.002
hidden_size = 64

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# Define the ConvNet model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the ConvNet model, criterion, and optimizer
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Training the ConvNet model
train_loss_arr, train_acc_arr, valid_loss_arr, valid_acc_arr = [], [], [], []
for epoch in range(max_iters):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_loss_arr.append(train_loss)
    train_acc_arr.append(train_acc)

    # Validation loop
    model.eval()
    correct_val = 0
    total_val = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(test_loader)
    val_acc = correct_val / total_val
    valid_loss_arr.append(val_loss)
    valid_acc_arr.append(val_acc)

    if epoch % 2 == 0:
        print("Epoch: {:02d} \t Train Loss: {:.2f} \t Train Accuracy : {:.2f}".format(
            epoch, train_loss, train_acc))

# Plotting Training Accuracy and Loss
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
plt.ylim(0, 2.5)
plt.title('Training and Validation Loss') 
plt.show()
