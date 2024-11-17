import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 3 * 3, 10),
            # nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 



# class MNISTNet(nn.Module):
#     def __init__(self):
#         super(MNISTNet, self).__init__()
#         self.conv1 = nn.Conv2d(1,4,kernel_size=3)
#         self.conv2 = nn.Conv2d(4,8,kernel_size=3)
#         self.conv3 = nn.Conv2d(8,8,kernel_size=3)
#         self.conv4 = nn.Conv2d(8,8,kernel_size=3)
#         self.fc1 = nn.Linear(3200,10)

#     def forward(self,x):
#         x = F.relu(self.conv1(x))#26
#         # x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))#24
#         # x = F.relu(F.max_pool2d(self.conv4(x), 2))
#         x = self.conv4(x)
#         # x = F.relu(self.conv3(x))#22
#         # x = F.relu(self.conv4(x))#20
#         x = x.view(-1,3200)
#         x = self.fc1(x)
#         x = F.log_softmax(x,1) # note we are applying along dimension1 bcoj in our case the dim of tensor before softmax would be (batch_size, num_classes) and we want outputs equal to num_classes i.e 10 prob values
#         return x

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 
    

# Accuracy after one epoch: 93.91% with parameters: 23274
# class MNISTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, kernel_size=3, stride=2),
#             # nn.Conv2d(32, 32, kernel_size=3),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32 * 2 * 2, 32),
#             nn.Linear(32, 10)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 

# Accuracy after one epoch: 94.05% with parameters: 23626
# class MNISTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, kernel_size=3),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32 * 3 * 3, 32),
#             nn.Linear(32, 10)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 

# Accuracy after one epoch: 93.84% with parameters: 16698
# class MNISTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 16, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32 * 3 * 3, 32),
#             nn.Linear(32, 10)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 

# Accuracy after one epoch: 93.30% with parameters: 25034
# class MNISTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(8, 16, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32 * 3 * 3, 64),
#             nn.Linear(64, 10)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 

# class MNISTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(8, 16, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32 * 1 * 1, 64),
#             nn.Linear(64, 10)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 

# class MNISTNet(nn.Module):
#     def __init__(self):
#         super(MNISTNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 3)
#         self.conv2 = nn.Conv2d(16,32,3)  # 1 input channel, 8 output channels, 3x3 kernel
#         self.conv3 = nn.Conv2d(32, 64, 3)  # 8 input channels, 16 output channels, 3x3 kernel
#         self.fc1 = nn.Linear(64 * 4 * 4, 64)  # Fully connected layer
#         self.fc2 = nn.Linear(64, 10)  # Output layer (10 classes)

#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 26x26 -> 13x13
#         x = F.relu(self.conv2(x)) # 11x11 -> 5x5
#         x = F.max_pool2d(F.relu(self.conv3(x)), 2) # 5x5 -> 3x3
#         x = x.view(-1, 64 * 4 * 4)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 