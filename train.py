import os
import torch
import torch.optim as optim
from torchvision.utils import save_image

from data_loader import get_dataloaders
from mapping import MessageEncoder
from generator import Generator
from discriminator import Discriminator
from extractor import Extractor
from loss import StegoLoss

def train_model(num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    os.makedirs("saved_images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    latent_dim = 256
    
    gen = Generator(latent_dim=latent_dim).to(device)
    disc = Discriminator().to(device)
    ext = Extractor(latent_dim=latent_dim).to(device)
    encoder = MessageEncoder(latent_dim=latent_dim, ecc_symbols=16)
    
    criterion = StegoLoss(lambda_msg=2.5) #we keep lambda_msg slightly lower so the Generator focuses on drawing numbers for image quality phasing. find the right balance
    
    # lr of 0.0002 is standard for GANs
    opt_gen_ext = optim.Adam(list(gen.parameters()) + list(ext.parameters()), lr=0.0002, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.999)) #0.0001/0.0002
    
    train_dl, _ = get_dataloaders(batch_size=64)

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_dl):
            
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator 
            opt_disc.zero_grad()
            
            random_binary = torch.randint(0, 2, (batch_size, latent_dim), dtype=torch.float32)
            secret_vectors = (random_binary * 2.0) - 1.0 
            secret_vectors = secret_vectors.to(device)
            
            fake_images = gen(secret_vectors)
            
            real_judgments = disc(real_images)
            fake_judgments = disc(fake_images.detach())
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            d_loss_real = criterion.adv_loss(real_judgments, real_labels)
            d_loss_fake = criterion.adv_loss(fake_judgments, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            d_loss.backward()
            opt_disc.step()

            # Train Generator & Extractor
            opt_gen_ext.zero_grad()
            
            fake_judgments_2 = disc(fake_images)
            extracted_vectors = ext(fake_images)
            
            total_g_loss, img_loss, data_loss = criterion(fake_judgments_2, real_labels, extracted_vectors, secret_vectors)
            
            total_g_loss.backward()
            opt_gen_ext.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Total Loss: {total_g_loss.item():.4f} | Data Loss: {data_loss.item():.4f}")
        
        # Save a grid of 25 fake images to see progress
        save_image(fake_images.data[:25], f"saved_images/epoch_new_{epoch+1}.png", nrow=5, normalize=True) 

    print("\nTraining complete. Saving AI weights.")
    torch.save(gen.state_dict(), "saved_models/generator.pth")
    torch.save(disc.state_dict(), "saved_models/discriminator.pth")
    torch.save(ext.state_dict(), "saved_models/extractor.pth")
    print("Models saved successfully in the 'saved_models' folder.")

if __name__ == "__main__":
    #check_training_loop()
    train_model(num_epochs=50) #change

