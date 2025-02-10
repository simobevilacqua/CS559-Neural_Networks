import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(10, 784))
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x0 = self.flatten(x)
        scores = torch.matmul(x0, self.centers.t()) - 0.5 * torch.sum(self.centers ** 2, 1).flatten()
        prob = self.softmax(20 * scores)
        reconstructed_x = torch.matmul(prob, self.centers)

        return reconstructed_x - x0

# Training
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        # Prediction and loss
        pred = model(X)
        loss = loss_fn(pred, torch.zeros_like(pred))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return loss

# Testing
def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, torch.zeros_like(pred)).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


def main(random=True):
    # Loading the datasets
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
        transform=ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size)

    # Initialize the model
    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize the centers
    basic_train_dataloader = DataLoader(train_data, batch_size=1)
    with torch.no_grad():
        if random:
            for i in range(10):
                model.centers[i, :] = torch.rand(784)
        else:
            i = 0
            for X, y in basic_train_dataloader:
                if y==i:
                    model.centers[i, :] = X.flatten()
                    i += 1
                    if i==10:
                        break
    
    # Training the model
    print(f"Training the model with centers initialized {'not ' if not random else ''}randomly\n")
    train_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
        train_losses.append(float(train_loss))
    
    # Output final cluster centers
    print("Final Cluster Centers:")
    print(model.centers)
    
    plt.figure(figsize=(10, 4))
    i = 0
    for center in model.centers:
        image = center.cpu().detach().numpy().reshape(28, 28)
        plt.subplot(2, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title(f'Center: {i}')
        i += 1
    plt.tight_layout()

    # Output confusion matrix
    conf_matrix = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for X, y in basic_train_dataloader:
            X = X.flatten()
            best_distance = float('inf')
            best_index = 0
            for j in range(10):
                dist = torch.norm(X - model.centers[j])
                if dist < best_distance:
                    best_distance = dist
                    best_index = j
            conf_matrix[y.item(), best_index] += 1

    print("Confusion Matrix:")
    print(conf_matrix.astype(int))

    # Plotting the loss
    plt.figure(figsize=(6, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss per Epoch')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main(False)
    main(True)