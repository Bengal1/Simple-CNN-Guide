# ----------------------------------------------------------------------
# Copyright (c) 2022, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
============================
SimpleCNN - MNIST Classifier
============================

This script defines and trains a simple Convolutional Neural Network (CNN) on the
MNIST digit dataset. It includes a clean training pipeline, evaluation routines,
and plotting functionality for loss trends.

Architecture:
- 2 Convolutional layers with ReLU.
- 2 Max pooling layers.
- 2 Fully connected layers.
- Regularization:
    - 2 Dropout.
    - 2 Batch Normalization.

Training Details:
- Loss Function: Cross-Entropy Loss is used to compute the loss between predicted
                logits and true labels.
- Optimizer: The Adam Optimizer is used to update the model's weights during training.
"""
__author__="Bengal1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


#--------------- Hyperparameters ---------------#
learning_rate = 1e-3
num_epochs = 10
batch_size = 256
num_class = 10
validation_split = 0.2  # 20% of training data for validation
#-------------- Config Parameters --------------#
input_channels = 1
conv1_out_channels = 32
conv2_out_channels = 64
conv_kernel_size = 5
pool_kernel_size = 2
pool_stride = 2
fc1_in = 64 * 4 * 4
fc2_in = 512
dropout1_rate = 0.45
dropout2_rate = 0.35


#--------------- Model Definition ---------------#
class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for handwritten digit classification on MNIST.

    This model consists of two convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool → Dropout),
    followed by two fully connected layers. It outputs raw logits and is intended
    to be used with `nn.CrossEntropyLoss`, which applies softmax internally.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        batch1 (nn.BatchNorm2d): Batch normalization after the first conv layer.
        max1 (nn.MaxPool2d): Max pooling after the first conv block.
        dropout1 (nn.Dropout): Dropout after the first pooling layer.

        conv2 (nn.Conv2d): Second convolutional layer.
        batch2 (nn.BatchNorm2d): Batch normalization after the second conv layer.
        max2 (nn.MaxPool2d): Max pooling after the second conv block.
        dropout2 (nn.Dropout): Dropout after the second pooling layer.

        fc1 (nn.Linear): First fully connected layer (dense).
        fc2 (nn.Linear): Output layer mapping to class logits.
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=conv1_out_channels,
                               kernel_size=conv_kernel_size)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,
                               out_channels=conv2_out_channels,
                               kernel_size=conv_kernel_size)

        # Max-Pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.max2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Fully-Connected layers
        self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc2_in)
        self.fc2 = nn.Linear(in_features=fc2_in, out_features=num_classes)

        # Dropout
        self.dropout1 = nn.Dropout(p=dropout1_rate)
        self.dropout2 = nn.Dropout(p=dropout2_rate)

        # Batch Normalization
        self.batch1 = nn.BatchNorm2d(num_features=conv1_out_channels)
        self.batch2 = nn.BatchNorm2d(num_features=conv2_out_channels)

    def forward(self, x):
        """
        Forward pass of the network.
        Note: CrossEntropyLoss handles softmax

        Args:
            x (torch.Tensor): Input batch of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)           # Convolution Layer 1
        x = F.relu(self.batch1(x))  # Batch Normalization + ReLU
        x = self.max1(x)            # Max Pooling
        x = self.dropout1(x)        # Dropout

        x = self.conv2(x)           # Convolution Layer 2
        x = F.relu(self.batch2(x))  # Batch Normalization + ReLU
        x = self.max2(x)            # Max Pooling
        x = self.dropout2(x)        # Dropout

        x = torch.flatten(x, start_dim=1)  # Flatten for FC layer
        x = F.relu(self.fc1(x))     # Fully Connected Layer 1 + ReLU
        x = self.fc2(x)             # Fully Connected Layer 2 (logits)
        return x


# --- Helper Function for Model Setup ---
def _setup_model_for_training(
        num_classes: int,
        lr: float
) -> tuple[nn.Module, nn.modules.loss, torch.optim.Optimizer, torch.device]:
    """
    Sets up the computational device, instantiates the CNN model,
    defines the loss function, and initializes the optimizer for training.

    Args:
        num_classes (int): The number of output classes for the model.
        lr (float): The learning rate for the optimizer.

    Returns:
        tuple[nn.Module, nn.modules.loss, torch.optim.Optimizer, torch.device]:
            A tuple containing:
            - The initialized SimpleCNN model.
            - The configured loss function (CrossEntropyLoss).
            - The initialized Adam optimizer.
            - The device (CPU or GPU).
    """
    # Set device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Instantiate the SimpleCNN model
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Initialize the Cross-Entropy Loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, loss_function, optimizer, device


#-------------- Training Utilities --------------#
def train_epoch(model: nn.Module,
                criterion: nn.modules.loss,
                optimizer: torch.optim,
                data_loader: DataLoader,
                device: torch.device) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        criterion (nn.modules.loss._Loss): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        data_loader (DataLoader): DataLoader for training data.
        device (torch.device): The computational device (CPU or GPU).

    Returns:
        tuple[float, float]: Tuple containing training accuracy (%) and average
                            training loss.
    """
    model.train()  # Training mode
    correct_train, total_train, total_train_loss = 0, 0, 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backpropagation
        loss.backward()
        # Update parameters
        optimizer.step()

        # Training accuracy calculation
        total_train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += predicted.eq(labels).sum().item()
        total_train += labels.size(0)

    epoch_accuracy = 100 * correct_train / total_train
    epoch_loss = total_train_loss / len(data_loader)

    return epoch_accuracy, epoch_loss


def evaluate_model(model: nn.Module,
                   criterion: nn.modules.loss,
                   data_loader: DataLoader,
                   device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on a validation or test set.

    Args:
        model (nn.Module): The neural network model.
        criterion (nn.modules.loss._Loss): The loss function.
        data_loader (DataLoader): DataLoader for validation/test data.
        device (torch.device): The computational device (CPU or GPU).

    Returns:
        tuple[float, float]: Tuple containing evaluation accuracy (%) and average loss.
    """
    model.eval()  # Evaluation mode
    total_eval_loss, correct_eval, total_eval = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Accuracy calculation
            total_eval_loss += loss.item()
            _, predicted = logits.max(1)
            correct_eval += predicted.eq(labels).sum().item()
            total_eval += labels.size(0)

    eval_accuracy = 100 * correct_eval / total_eval
    eval_loss = total_eval_loss / len(data_loader)

    return eval_accuracy, eval_loss


def train_model(model: nn.Module,
                criterion: nn.modules.loss,
                optimizer: torch.optim,
                training_loader: DataLoader,
                validation_loader: DataLoader,
                device: torch.device) -> tuple[list[float], list[float]]:
    """
    Trains the CNN model over multiple epochs using training and validation data.

    Args:
        model (nn.Module): The CNN model to train.
        criterion (nn.modules.loss._Loss): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        training_loader (DataLoader): DataLoader providing training data.
        validation_loader (DataLoader): DataLoader providing validation data.
        device (torch.device): The computational device (CPU or GPU).

    Returns:
        tuple[list[float], list[float]]: Lists of training and validation losses
                                        per epoch.
    """
    train_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        train_accuracy, epoch_train_loss = train_epoch(model,
                                                       criterion,
                                                       optimizer,
                                                       training_loader,
                                                       device)
        train_loss.append(epoch_train_loss)

        validation_accuracy, epoch_validation_loss = evaluate_model(model,
                                                                    criterion,
                                                                    validation_loader,
                                                                    device)
        validation_loss.append(epoch_validation_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss[epoch]:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}% | Validation Loss:"
              f" {validation_loss[epoch]:.4f}, Validation Accuracy:"
              f" {validation_accuracy:.2f}%")

    return train_loss, validation_loss


#-------------- Visualization -------------------#
def plot_training_losses(train_loss_epochs: list[float],
                         validation_loss_epochs: list[float]):
    """
    Plots training and validation loss over epochs.

    Args:
        train_loss_epochs (list[float]): List of training loss values per epoch.
        validation_loss_epochs (list[float]): List of validation loss values per epoch.
    """
    eps = range(1, len(train_loss_epochs) + 1)
    # --- Plotting Configuration ---
    plt.figure(figsize=(10, 5))
    plt.plot(eps, train_loss_epochs, linestyle='-', color='#1f77b4',
             label='Train Loss', linewidth=2)
    plt.plot(eps, validation_loss_epochs, linestyle='-', color='#d62728',
             label='Validation Loss', linewidth=2)
    # --- Chart Customization ---
    plt.title("Training & Validation Loss Over Epochs", fontsize=16,
              fontweight='bold')
    plt.xticks(eps)  # This ensures that xticks are integers
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # --- Display Plot ---
    plt.show()


#----------------- Data Loading -----------------#
def get_mnist_dataloaders(samples_per_batch: int,
                          train_validation_split: float
                          ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the MNIST dataset and creates DataLoader objects for training,
    validation, and testing.

    Args:
        samples_per_batch (int): The batch size for DataLoaders.
        train_validation_split (float): The fraction of the training data to use
                                  for validation (e.g., 0.2 for 20%).

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: A tuple containing
                                (training_loader, validation_loader, testing_loader).
    """
    # Load MNIST dataset
    full_train_dataset = datasets.MNIST(root='./data', train=True,
                                        download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=ToTensor())

    # Split into training (80%) and validation (20%)
    train_dataset, val_dataset = random_split(full_train_dataset,
                                              [1 - train_validation_split,
                                               train_validation_split])

    # Create DataLoader for training, validation and test datasets.
    training_loader = DataLoader(train_dataset,
                                 batch_size=samples_per_batch, shuffle=True)
    validation_loader = DataLoader(val_dataset,
                                   batch_size=samples_per_batch, shuffle=False)
    testing_loader = DataLoader(test_dataset,
                                batch_size=samples_per_batch, shuffle=False)

    return training_loader, validation_loader, testing_loader


#------------------ Main Entry ------------------#
if __name__ == "__main__":
    # Initialize model, loss function and optimizer
    cnn_model, loss_fn, adam_optimizer, h_device = _setup_model_for_training(
                                                            num_class,learning_rate)
    # Initialize MNIST data loaders
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
                                                    batch_size, validation_split)
    # Train & Validation
    train_losses, validation_losses = train_model(cnn_model, loss_fn, adam_optimizer,
                                                  train_loader, val_loader, h_device)
    # Test
    test_accuracy, test_loss = evaluate_model(cnn_model, loss_fn,
                                              test_loader, h_device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    # Plot Loss
    plot_training_losses(train_losses, validation_losses)