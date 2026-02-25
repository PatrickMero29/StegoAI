import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 1),
            
            # probability between 0.0 and 1.0
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        
        return validity

if __name__ == "__main__":
    
    disc = Discriminator()
    print("Discriminator initialized successfully.\n")

    # Temp testing inputs
    batch_size = 4
    fake_images = torch.randn(batch_size, 1, 28, 28)
    print(f"Input Image Shape (Batch Size, Channels, H, W): {fake_images.shape}")

    # forward pass
    with torch.no_grad():
        judgments = disc(fake_images)

    print(f"Output Judgment Shape: {judgments.shape}")
    print(f"Scores (Probability of being Real):\n{judgments.squeeze().numpy()}\n")

    if judgments.shape == (batch_size, 1):
        print("SUCCESS: The output shape is correct. Each image has a single score.")
    else:
        print("Error: The output shape is incorrect. Expected (batch_size, 1).")