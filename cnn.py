import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class CNNModel(nn.Module):
    """Two-layer CNN for MNIST digit classification."""

    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Reshape: (batch, 32, 4, 4) → (batch, 512)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


def train(model, train_loader, test_loader, num_epochs, learning_rate, device):
    """Train the CNN and evaluate every 500 iterations."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    iter_count = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax → cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter_count += 1

            if iter_count % 500 == 0:
                accuracy = evaluate(model, test_loader, device)
                print(
                    f"Iteration: {iter_count}. Loss: {loss.item():.4f}. "
                    f"Accuracy: {accuracy:.2f}%"
                )


def evaluate(model, test_loader, device):
    """Evaluate model accuracy on the test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def main():
    # ── STEP 1: Load dataset ──────────────────────────────────────────
    train_dataset = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    # ── STEP 2: Make dataset iterable ─────────────────────────────────
    batch_size = 100
    n_iters = 3000
    num_epochs = int(n_iters / (len(train_dataset) / batch_size))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # ── STEP 3: Instantiate model ─────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)

    # ── STEP 4: Train ─────────────────────────────────────────────────
    learning_rate = 0.01
    train(model, train_loader, test_loader, num_epochs, learning_rate, device)


if __name__ == "__main__":
    main()
