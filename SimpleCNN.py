import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Hyper Parameters #
learning_rate = 1e-3
num_epochs = 10
batch_size = 256
num_class = 10
validation_split = 0.2  # 20% of training data for validation


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for MNIST classification.

    Architecture:
    - 2 Convolutional layers with ReLU and Batch Normalization
    - 2 Max Pooling layers
    - 2 Dropout layers for regularization
    - 2 Fully Connected (FC) layers
    - No explicit Softmax (handled by CrossEntropyLoss)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        # Max pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(p=0.45)
        self.dropout2 = nn.Dropout(p=0.35)

        # Batch Normalization layers
        self.batch1 = nn.BatchNorm2d(num_features=32)
        self.batch2 = nn.BatchNorm2d(num_features=64)

        # Fully Connected layers
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the network.
        Note: CrossEntropyLoss handles softmax

        Args:
            x (torch.Tensor): Input batch of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)                   # Convolution Layer 1
        x = F.relu(self.batch1(x))          # Batch Normalization + ReLU
        x = self.max1(x)                    # Max Pooling
        x = self.dropout1(x)                # Dropout

        x = self.conv2(x)                   # Convolution Layer 2
        x = F.relu(self.batch2(x))          # Batch Normalization + ReLU
        x = self.max2(x)                    # Max Pooling
        x = self.dropout2(x)                # Dropout

        x = torch.flatten(x, start_dim=1)   # Flatten for FC layer
        x = F.relu(self.fc1(x))             # Fully Connected Layer 1 + ReLU
        x = self.fc2(x)                     # Fully Connected Layer 2 (logits)
        return x


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# Load MNIST dataset #
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

# Split into training (80%) and validation (20%)
train_dataset, val_dataset = random_split(full_train_dataset,
                                        [1-validation_split, validation_split])

# Create DataLoader for training, validation and test datasets.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = SimpleCNN(num_classes=num_class).to(device)

# Loss & Optimization #
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_loss = []
validation_loss = []

# Train & Validation #
for epoch in range(num_epochs):
    model.train()  # training mode
    correct_train, total_train, total_train_loss = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()               # Reset gradients
        outputs = model(inputs)             # Forward pass
        loss = criterion(outputs, labels)   # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update parameters

        # Training accuracy calculation
        total_train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += predicted.eq(labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = 100 * correct_train / total_train
    train_loss.append(total_train_loss / len(train_loader))

    # Validation
    model.eval()  # Set to evaluation mode
    total_val_loss, correct_val, total_val = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Validation accuracy calculation
            total_val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_val += predicted.eq(labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = 100 * correct_val / total_val
    validation_loss.append(total_val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}: "
          f"Train Loss: {train_loss[epoch]:.4f}, Train Accuracy: {train_accuracy:.2f}% | "
          f"Validation Loss: {validation_loss[epoch]:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


# Test #
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
        # test accuracy calculation
        _, predicted = outputs.max(1)
        correct_test += predicted.eq(labels).sum().item()
        total_test += labels.size(0)

test_accuracy = 100 * correct_test / total_test
print(f"\nTest Accuracy: {test_accuracy:.2f}%")


# Plot Loss #
eps = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))
plt.plot(eps, train_loss, linestyle='-', color='#1f77b4', label='Train Loss', linewidth=2)
plt.plot(eps, validation_loss, linestyle='-', color='#d62728', label='Validation Loss', linewidth=2)

plt.title("Training & Validation Loss Over Epochs", fontsize=16, fontweight='bold')
plt.xticks(eps) # This ensures that xticks are integers
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()