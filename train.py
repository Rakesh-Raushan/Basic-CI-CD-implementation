import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from network import MNISTNet
import tqdm

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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for one epoch
    accuracy = train_one_epoch(model, device, train_loader, optimizer, criterion)
    print(f"Accuracy after one epoch: {accuracy:.2f}% with parameters: {model.count_parameters()}")
    
    # Save model and metrics
    torch.save(model.state_dict(), 'mnist_model.pth')
    with open('training_metrics.txt', 'w') as f:
        f.write(f"{accuracy}")
    
    return accuracy

if __name__ == "__main__":
    main() 