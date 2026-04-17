import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # (3, 64, 64) 
            # Downsample to 32x32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            
            # The flattened size is 512 channels * 4 height * 4 width = 8192
            nn.Linear(512 * 4 * 4, 1),
            
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

if __name__ == "__main__":
    disc = Discriminator()
    print("DCGAN Discriminator initialized successfully.\n")

    batch_size = 4
    fake_images = torch.randn(batch_size, 3, 64, 64)
    print(f"Input Image Shape: {fake_images.shape}")

    with torch.no_grad():
        judgments = disc(fake_images)

    print(f"Output Judgment Shape: {judgments.shape}")
    print(f"Scores (Probability of being Real):\n{judgments.squeeze().numpy()}\n")

    if judgments.shape == (batch_size, 1):
        print("SUCCESS: The output shape is correct.")
    else:
        print("Error: The output shape is incorrect. Expected (batch_size, 1).")