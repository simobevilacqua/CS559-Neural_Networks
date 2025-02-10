import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.1
num_epochs = 10
batch_size = 64
weight_decay = 1e-4

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network architectures
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Tensor size: 1x28x28

            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            # Tensor size: 20x25x25

            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2),
            nn.ReLU(),
            # Tensor size: 20x11x11

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            # Tensor size: 20x5x5

            nn.Flatten(),
            # Tensor size: 500

            nn.Linear(500, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Dropout(0.1),
            # Tensor size: 250

            nn.Linear(250, 10)
            # Tensor size: 10
        )
        self.decoder = nn.Sequential(
            # Tensor size: 10

            nn.Linear(10, 360),
            nn.ReLU(),
            nn.BatchNorm1d(360),
            nn.Dropout(0.1),
            # Tensor size: 360

            nn.Linear(360, 720),
            nn.ReLU(),
            # Tensor size: 720

            nn.Unflatten(1, (20, 6, 6)),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            # Tensor size: 20x6x6

            nn.Upsample(scale_factor=2, mode='bicubic'),
            # Tensor size: 20x12x12

            nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            # Tensor size: 20x24x24

            nn.ConvTranspose2d(20, 1, kernel_size=4, stride=1, output_padding=0),
            nn.Sigmoid()
            # Tensor size: 1x28x28
        )
        self.flatten = nn.Flatten()

    def forward(self, x, enc_mode=1):
        z = self.encoder(x)
        z = z.to(device)
        z2 = enc_mode * z + (2 - enc_mode) * torch.randn(z.shape, device=device)
        f = self.decoder(z2)

        reconstruction_error = self.flatten(f - x)
        output = torch.cat((z, reconstruction_error), dim=1)

        return output

class AutoencoderAdvanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Tensor size: 1x28x28

            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            # Tensor size: 64x25x25

            nn.Conv2d(in_channels=64, out_channels=200, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            # Tensor size: 200x11x11

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(200),
            nn.Dropout(0.1),
            # Tensor size: 200x5x5

            nn.Flatten(),
            # Tensor size: 5000

            nn.Linear(5000, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            # Tensor size: 512

            nn.Linear(512, 10)
            # Tensor size: 10
        )
        self.decoder = nn.Sequential(
            # Tensor size: 10

            nn.Linear(10, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            # Tensor size: 512

            nn.Linear(512, 7200),
            nn.LeakyReLU(),
            # Tensor size: 7200

            nn.Unflatten(1, (200, 6, 6)),
            nn.BatchNorm2d(200),
            nn.Dropout(0.1),
            # Tensor size: 200x6x6

            nn.Upsample(scale_factor=2, mode='bicubic'),
            # Tensor size: 200x12x12

            nn.ConvTranspose2d(200, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            # Tensor size: 64x24x24

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=1, output_padding=0),
            nn.Sigmoid()
            # Tensor size: 1x28x28
        )
        self.flatten = nn.Flatten()

    def forward(self, x, enc_mode=1):
        z = self.encoder(x)
        z = z.to(device)
        z2 = enc_mode * z + (2 - enc_mode) * torch.randn(z.shape, device=device)
        f = self.decoder(z2)
        
        reconstruction_error = self.flatten(f - x)
        output = torch.cat((z, reconstruction_error), dim=1)

        return output

# Training
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        # Prediction and loss
        X = X.to(device)
        pred = model(X)
        z, recon_error = pred[:, :10], pred[:, 10:]
        loss = torch.mean(z ** 2) + loss_fn(recon_error, torch.zeros_like(recon_error))

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
    test_loss = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)
            recon_error = pred[:, 10:]
            test_loss += loss_fn(recon_error, torch.zeros_like(recon_error)).item()

    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

    return test_loss

def execute_model(model, loss_fn, optimizer, train_loader, test_loader):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_losses.append(float(train_loop(train_loader, model, loss_fn, optimizer)))
        test_losses.append(float(test_loop(test_loader, model, loss_fn)))

    # Generating new images using the trained autoencoder
    plt.figure(figsize=(12, 10))
    for i in range(20):
        x = model(torch.zeros(1, 1, 28, 28).to(device), 0)
        imgX = x[:, 10:].reshape(28, 28).detach().to("cpu")
        plt.subplot(4, 5, i + 1)
        plt.imshow(imgX, cmap='grey')
        plt.axis('off')
        plt.title(f"Image {i + 1}")
    plt.tight_layout()

    # Plotting the training loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss per Epoch')
    plt.legend()

    # Plotting the testing loss
    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Testing loss per Epoch')
    plt.legend()


def main():
    global learning_rate, num_epochs, batch_size

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
        transform=ToTensor(),
    )

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    # Base architecture
    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    execute_model(model, loss_fn, optimizer, train_loader, test_loader)

    # Advanced architecture
    learning_rate = 0.05
    num_epochs = 20
    batch_size = 32
    model = AutoencoderAdvanced().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    execute_model(model, loss_fn, optimizer, train_loader, test_loader)

    plt.show()


if __name__ == "__main__":
    main()