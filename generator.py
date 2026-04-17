import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=512):
        super(Generator, self).__init__()
        
        self.project = nn.Linear(latent_dim, 512 * 4 * 4)
        
        self.main = nn.Sequential(
            # Input State: [Batch, 512, 4, 4]
            # ConvTranspose formula for size: (size - 1) * stride - 2 * padding + kernel_size
            # (4 - 1)*2 - 2*1 + 4 = 8. Spatial size becomes 8x8.
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # State: [Batch, 256, 8, 8]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # State: [Batch, 128, 16, 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # State: [Batch, 64, 32, 32]
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            
            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z)
        x = x.view(x.size(0), 512, 4, 4) 
        img = self.main(x)
        
        return img
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    latent_dim = 512
    gen = Generator(latent_dim=latent_dim)
    print("DCGAN Generator initialized successfully.\n")

    batch_size = 4
    z = torch.randn(batch_size, latent_dim)
    print(f"Input Vector Shape: {z.shape}")

    with torch.no_grad():
        fake_images = gen(z)

    print(f"Output Image Shape: {fake_images.shape}")
    print(f"Min pixel value (Should be >= -1.0): {fake_images.min().item():.4f}")
    print(f"Max pixel value (Should be <= 1.0): {fake_images.max().item():.4f}\n")

    if fake_images.shape == (batch_size, 3, 64, 64):
        print("Success: The output shape is correct. Each image is 64x64 RGB.")
    else:
        print("Error: The output shape is incorrect. Check the convolution math.")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        img_unnorm = fake_images[i] * 0.5 + 0.5
        img_np = np.transpose(img_unnorm.numpy(), (1, 2, 0))
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Untrained RGB {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()