import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def train_model(model, optimizer, num_epochs, learning_rate, batch_size):
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to CUDA if available
    model.to(device)

    # Define datasets and dataloaders
    def get_datasets():
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

        return train_dataset, test_dataset

    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and learning rate
    criterion = nn.CrossEntropyLoss()
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_accuracy = correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_accuracy = correct_val / total_val
        avg_val_loss = val_loss / len(test_loader)

        # Print training and validation accuracy
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {100 * train_accuracy:.2f}%, '
              f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {100 * val_accuracy:.2f}')

# Example usage:
# model = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3, 4, 6, 4], widths=[256, 512, 1024, 2048], num_classes=10)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# train_model(model, optimizer, num_epochs=10, learning_rate=0.01, batch_size=64)
