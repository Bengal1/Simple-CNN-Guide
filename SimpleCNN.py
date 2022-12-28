import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Hyper Parameters #
learning_rate = 1e-3
num_epochs = 20
batch_size = 256
num_class = 10

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# Data #
"""
MNIST datasets from torchvision.datasets
"""
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SimpleCNN(nn.Module):
    """
    defining simple CNN with 2 convolution layers, 2 pooling layers and 2 fully-connected layers
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Network's Layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.max1 = nn.MaxPool2d(2, stride=2)
        self.max2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_class)
        # Dropout
        self.dropout1 = nn.Dropout(0.45)
        self.dropout2 = nn.Dropout(0.35)
        # Batch Normalization
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)                   # Convolution layer
        x = F.relu(self.batch1(x))          # Batch Normalization + ReLU
        x = self.max1(x)                    # Max-pool layer
        x = self.dropout1(x)                # Dropout

        x = self.conv2(x)                   # Convolution layer
        x = F.relu(self.batch2(x))          # Batch Normalization + ReLU
        x = self.max2(x)                    # Max-pool layer
        x = self.dropout2(x)                # Dropout

        x = x.reshape(-1, 1024)             # Flatten
        x = F.relu(self.fc1(x))             # Fully-connected layer + ReLU
        x = F.softmax(self.fc2(x), dim=0)   # Fully-connected layer + Softmax
        return x


net = SimpleCNN()
net.to(device)

# Loss & Optimization #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Train #
for epoch in range(num_epochs):
    correct_train = 0
    total_train = 0

    net.train()  # training mode
    print('epoch #{}' .format(epoch+1))
    for i, (inputs, labels) in enumerate(train_loader):
        inputs.to(device), labels.to(device)

        # set gradient's parameter to zero
        optimizer.zero_grad()

        # forward pass & back propagation
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    print('Train Accuracy: %0.3f' % ((100 * correct_train) / total_train))


# Test #
net.eval()
correct_test = 0
total_test = 0

for i, (inputs, labels) in enumerate(test_loader):
    inputs.to(device), labels.to(device)

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    
    # test accuracy calculation
    _, predicted = torch.max(outputs.data, 1)
    correct_test += (predicted == labels).sum().item()
    total_test += labels.size(0)

print('\nTest Accuracy: %0.3f' % ((100 * correct_test) / total_test))
