import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  # 1 input channel, 8 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(8, 16, 3)  # 8 input channels, 16 output channels, 3x3 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 32)  # Fully connected layer
        self.fc2 = nn.Linear(32, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Conv -> ReLU -> MaxPool
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Conv -> ReLU -> MaxPool
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 