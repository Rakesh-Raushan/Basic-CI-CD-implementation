import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def show_transformed_images(dataset, num_images=5, cols=5):
    rows = int(np.ceil(num_images / cols))
    figure = plt.figure(figsize=(2*cols, 2*rows))
    
    for i in range(num_images):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        
        figure.add_subplot(rows, cols, i + 1)
        plt.title(f'Label: {label}')
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.close()

if __name__ == "__main__":
    # Define augmentations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset with augmentations
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # Show some augmented images
    show_transformed_images(dataset, num_images=10)
    print("Augmented samples saved as 'augmented_samples.png'") 