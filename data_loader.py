import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_dataloaders(batch_size=64):
    # Add normalization so pixels are between -1.0 and 1.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # load data
    test_dataset = datasets.MNIST(
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
    print("Data loaded successfully!")
    
    train_features, train_labels = next(iter(train_dl))
    
    fig, axes = plt.subplots(1, 10, figsize=(10, 3))
    for i in range(10):
        # We have to un-normalize the image just to display it properly in pyplot
        img = train_features[i].squeeze() * 0.5 + 0.5 
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {train_labels[i].item()}")
        axes[i].axis('off')
    
    plt.show()