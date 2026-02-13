import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    train_dataset = datasets.MNIST(
        root="data", 
        train=True, 
        download=True, 
        transform=ToTensor()
    )
    
    # Download and load the testing data
    test_dataset = datasets.MNIST(
        root="data", 
        train=False, 
        download=True, 
        transform=ToTensor()
    )
    
    # Wrap datasets in DataLoaders to pass samples in minibatches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_dl, test_dl = get_dataloaders()
    print("Data loaded successfully!")
    
    train_features, train_labels = next(iter(train_dl))
    
    # Plot first 4 images from batch
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        # .squeeze() removes the '1' color channel so matplotlib understands it's a flat 28x28 grid
        axes[i].imshow(train_features[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {train_labels[i].item()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()