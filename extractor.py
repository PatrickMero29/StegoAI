import torch
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, latent_dim=256):
        super(Extractor, self).__init__()
        
        self.model = nn.Sequential(
            # Takes the [1, 28, 28] image and creates 32 feature maps
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            
            # Takes the 32 maps and turns them into 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Flatten(), # 64 channels * 7 height * 7 width = 3136
            
            # Dense layer to shrink the features
            nn.Linear(3136, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            # 256 numbers to match latent vector size
            nn.Linear(512, latent_dim),
            
            # output guesses strictly between -1.0 and 1.0, 
            nn.Tanh()
        )

    def forward(self, img):
        extracted_message = self.model(img)
        return extracted_message

if __name__ == "__main__":
    latent_dim = 256
    ext = Extractor(latent_dim=latent_dim)
    print("Extractor initialized successfully.\n")

    batch_size = 4
    fake_images = torch.randn(batch_size, 1, 28, 28)
    print(f"Input Image Shape: {fake_images.shape}")

    with torch.no_grad():
        recovered_vector = ext(fake_images)

    print(f"Output Vector Shape: {recovered_vector.shape}")
    print(f"Min value (Should be >= -1.0): {recovered_vector.min().item():.4f}")
    print(f"Max value (Should be <= 1.0): {recovered_vector.max().item():.4f}\n")

    if recovered_vector.shape == (batch_size, latent_dim):
        print("Extractor matrix math is perfectly aligned. Output shape is correct.")
    else:
        print("The output shape is incorrect.")