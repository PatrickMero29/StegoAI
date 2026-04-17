import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    train_dataset = datasets.CIFAR10(
        root="data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root="data", 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders()
    
    train_features, train_labels = next(iter(train_dl))
    
    print(f"New Image Batch Shape: {train_features.shape}") # Should be [64, 3, 64, 64]
    
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        img = train_features[i] * 0.5 + 0.5 
        
        img_np = np.transpose(img.numpy(), (1, 2, 0))
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Class: {train_labels[i].item()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()