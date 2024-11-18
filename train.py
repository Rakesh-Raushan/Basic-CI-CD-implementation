import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from network import MNISTNet
import tqdm
import random
import argparse

class RandomApplyTransform:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img

def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total

def get_transforms(use_augmentation=False):
    if use_augmentation:
        return transforms.Compose([
            transforms.ToTensor(),
            RandomApplyTransform(transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.RandomPerspective(distortion_scale=0.2),
            ]), p=0.5),  # 50% chance of applying augmentations
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

def main(use_augmentation=False):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get transforms based on augmentation flag
    train_transform = get_transforms(use_augmentation)
    
    # Load data with augmentations
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer settings with fixed learning rate
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001,  # Back to standard learning rate
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Train for one epoch
    accuracy = train_one_epoch(model, device, train_loader, optimizer, criterion)
    print(f"Accuracy after one epoch: {accuracy:.2f}% with parameters: {model.count_parameters()}")
    print(f"Augmentations {'enabled' if use_augmentation else 'disabled'}")
    
    # Save model and metrics
    torch.save(model.state_dict(), 'mnist_model.pth')
    with open('training_metrics.txt', 'w') as f:
        f.write(f"{accuracy}")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    args = parser.parse_args()
    
    main(use_augmentation=args.augment) 