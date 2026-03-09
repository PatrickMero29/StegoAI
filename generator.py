import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=256):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            #256-bit vector and expand 
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            #Expand to 512 neurons
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            #Expand to 1024 neurons
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            #(1 color channel * 28 * 28)
            nn.Linear(1024, 1 * 28 * 28),
            
            nn.Tanh()
        )

    def forward(self, z):
        #z = the latent vector (seed + message)
        img_flat = self.model(z)
        
        #[batch_size, channels, height, width]
        img = img_flat.view(img_flat.size(0), 1, 28, 28)
        
        return img
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    latent_dim = 256
    gen = Generator(latent_dim=latent_dim)
    print("Generator initialized successfully.\n")

    # Testing inputs
    batch_size = 4
    z = torch.randn(batch_size, latent_dim)
    print(f"Input Vector Shape (Batch Size, Latent Dim): {z.shape}")

    # torch.no_grad() for testing, not training
    with torch.no_grad():
        fake_images = gen(z)

    print(f"Output Image Shape: {fake_images.shape}")
    print(f"Min pixel value (Should be >= -1.0): {fake_images.min().item():.4f}")
    print(f"Max pixel value (Should be <= 1.0): {fake_images.max().item():.4f}\n")

    if fake_images.shape == (batch_size, 1, 28, 28):
        print("Success: The output shape is correct. Each image has 1 channel and is 28x28 pixels.")
    else:
        print("Error: The output shape is incorrect. Expected (batch_size, 1, 28, 28).")

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        # .squeeze() to remove the color channel, matplotlib can read it
        img_np = fake_images[i].squeeze().numpy()
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f"Untrained {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()