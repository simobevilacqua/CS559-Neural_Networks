import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
learning_rate = 0.001
num_epochs = 20
batch_size = 64
weight_decay = 1e-4
dropout_first_layer = 0.25
dropout_second_layer = 0.25

# Network structure
input_channels_first_layer = 1
output_channels_first_layer = 20
kernel_size_first_layer = 4
stride_first_layer = 1
input_channels_second_layer = 20
output_channels_second_layer = 20
kernel_size_second_layer = 4
stride_second_layer = 2
kernel_size_max_pooling = 2
stride_max_pooling = 2
# Output after the first convolutional layer: 20x25x25
# Output after the second convolutional layer: 20x11x11
# Output after the max pooling layer: 20x5x5
input_size = 20*5*5 #500
hidden_size = 250
output_size = 10

# Define networks
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_CNN = nn.Sequential(
            nn.Conv2d(in_channels=input_channels_first_layer, out_channels=output_channels_first_layer, kernel_size=kernel_size_first_layer, stride=stride_first_layer),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels_second_layer, out_channels=output_channels_second_layer, kernel_size=kernel_size_second_layer, stride=stride_second_layer),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_max_pooling, stride=stride_max_pooling)
        )
        self.model_fully_connected_NN = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model_CNN(x)
        x = x.view(x.size(0), -1)
        logits = self.model_fully_connected_NN(x)
        return logits


class ConvolutionalNeuralNetworkAdvanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_CNN = nn.Sequential(
            nn.Conv2d(in_channels=input_channels_first_layer, out_channels=output_channels_first_layer, kernel_size=kernel_size_first_layer, stride=stride_first_layer),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels_second_layer, out_channels=output_channels_second_layer, kernel_size=kernel_size_second_layer, stride=stride_second_layer),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_max_pooling, stride=stride_max_pooling)
        )
        self.model_fully_connected_NN = nn.Sequential(
            nn.Dropout(dropout_first_layer),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_second_layer),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model_CNN(x)
        x = x.view(x.size(0), -1)
        logits = self.model_fully_connected_NN(x)
        return logits


class ConvolutionalNeuralNetworkFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_CNN = nn.Sequential(
            nn.Conv2d(in_channels=input_channels_first_layer, out_channels=output_channels_first_layer, kernel_size=kernel_size_first_layer, stride=stride_first_layer),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels_first_layer),
            nn.Conv2d(in_channels=input_channels_second_layer, out_channels=output_channels_second_layer, kernel_size=kernel_size_second_layer, stride=stride_second_layer),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels_second_layer),
            nn.MaxPool2d(kernel_size=kernel_size_max_pooling, stride=stride_max_pooling)
        )
        self.model_fully_connected_NN = nn.Sequential(
            nn.Dropout(dropout_first_layer),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_second_layer),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model_CNN(x)
        x = x.view(x.size(0), -1)
        logits = self.model_fully_connected_NN(x)
        return logits


class EarlyStopping:
    def __init__(self, limit=3):
        self.limit = limit
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.limit:
                self.early_stop = True


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # Avoid computing gradients
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return accuracy


def execute_model(model, loss_fn, optimizer, train_loader, test_loader, apply_early_stopping=False):
    if apply_early_stopping:
        early_stopping = EarlyStopping()
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        loss = train_loop(train_loader, model, loss_fn, optimizer)
        accuracy = test_loop(test_loader, model, loss_fn)
        losses.append(float(loss))
        accuracies.append(float(accuracy))

        # Early stopping check
        if apply_early_stopping:
            early_stopping(loss)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

    # Plotting the loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss per Epoch')
    plt.legend()

    # Plotting the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy per Epoch')
    plt.legend()


def main():
    # Loading the dataset
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create DataLoader for training and testing datasets
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Select best device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Run the model with basic configuration (Point b)
    model = ConvolutionalNeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    execute_model(model, loss_fn, optimizer, train_loader, test_loader)

    # Run the model with advanced configuration (Point c)
    model = ConvolutionalNeuralNetworkAdvanced().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    execute_model(model, loss_fn, optimizer, train_loader, test_loader)

    # Run the model with the final configuration (Point d)
    model = ConvolutionalNeuralNetworkFinal().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # execute_model(model, loss_fn, optimizer, train_loader, test_loader, False)
    execute_model(model, loss_fn, optimizer, train_loader, test_loader, True)

    plt.show()


if __name__ == "__main__":
    main()