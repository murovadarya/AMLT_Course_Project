import torch
from torch import nns
from torch import Tensor
from typing import List
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm = nn.BatchNorm2d,
        act = nn.ReLU,
        **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features),
            act(),
        )

class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        reduction: int = 4,
        stride: int = 1,
    ):
        super().__init__()
        reduced_features = out_features // reduction
        self.block = nn.Sequential(
            ConvNormAct(
                in_features, reduced_features, kernel_size=1, stride=stride, bias=False
            ),
            ConvNormAct(reduced_features, reduced_features, kernel_size=3, bias=False),
            ConvNormAct(reduced_features, out_features, kernel_size=1, bias=False, act=nn.Identity),
        )
        self.shortcut = (
            nn.Sequential(
                ConvNormAct(
                    in_features, out_features, kernel_size=1, stride=stride, bias=False
                )
            )
            if in_features != out_features
            else nn.Identity()
        )

        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        res = self.shortcut(res)
        x += res
        x = self.act(x)
        return x

class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, stride: int = 2, **kwargs
    ):
        super().__init__(
            BottleNeckBlock(in_features, out_features, stride=stride, **kwargs),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth - 1)
            ],
        )

class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            ConvNormAct(
                in_features, out_features, kernel_size=3, stride=1  # Adjust kernel_size and stride
            ),
        )
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            ConvNormAct(
                in_features, out_features, kernel_size=3, stride=1  # Adjust kernel_size and stride
            ),
        )

class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        num_classes: int = 10  # Adjust for CIFAR-10
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))

        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], stride=1),
                *[
                    ConvNexStage(in_features, out_features, depth)
                    for (in_features, out_features), depth in zip(
                        in_out_widths, depths[1:]
                    )
                ],
            ]
        )

        # Global Average Pooling (GAP)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        # Global Average Pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer
        x = self.fc(x)

        return x

encoder = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3, 4, 6, 4], widths=[256, 512, 1024, 2048], num_classes=10)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])


batch_size = 64

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# CIFAR-10 Test Dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate model, loss function, and optimizer
model = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3, 4, 6, 4], widths=[256, 512, 1024, 2048], num_classes=10)
model.to(device)  # Move the model to CUDA if available
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/ {num_epochs} - Training'):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to CUDA
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
        for inputs, labels in tqdm(test_loader, desc=f'Epoch {epoch + 1}/ {num_epochs} - Validation'):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to CUDA
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
          f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {100 * val_accuracy:.2f}%')