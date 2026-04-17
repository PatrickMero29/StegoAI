import torch
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, latent_dim=512):
        super(Extractor, self).__init__()
        
        self.model = nn.Sequential(
            # Input is (3, 64, 64)
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
            
            # Flattened size is 512 * 4 * 4 = 8192
            nn.Linear(8192, latent_dim),
            
            nn.Tanh()
        )

    def forward(self, img):
        extracted_message = self.model(img)
        return extracted_message

if __name__ == "__main__":
    latent_dim = 512
    ext = Extractor(latent_dim=latent_dim)
    print("DCGAN Extractor initialized successfully.\n")

    batch_size = 4
    fake_images = torch.randn(batch_size, 3, 64, 64)
    print(f"Input Image Shape: {fake_images.shape}")

    with torch.no_grad():
        recovered_vector = ext(fake_images)

    print(f"Output Vector Shape: {recovered_vector.shape}")
    print(f"Min value (Should be >= -1.0): {recovered_vector.min().item():.4f}")
    print(f"Max value (Should be <= 1.0): {recovered_vector.max().item():.4f}\n")

    if recovered_vector.shape == (batch_size, latent_dim):
        print("SUCCESS: Matrix math is perfectly aligned. Output shape is correct.")
    else:
        print("Error: The output shape is incorrect.")