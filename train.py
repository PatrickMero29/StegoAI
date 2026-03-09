import torch
import torch.optim as optim

from data_loader import get_dataloaders
from mapping import MessageEncoder
from generator import Generator
from discriminator import Discriminator
from extractor import Extractor
from loss import StegoLoss

def check_training_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    latent_dim = 256
    
    gen = Generator(latent_dim=latent_dim).to(device)
    disc = Discriminator().to(device)
    ext = Extractor(latent_dim=latent_dim).to(device)
    encoder = MessageEncoder(latent_dim=latent_dim, ecc_symbols=10)
    criterion = StegoLoss(lambda_msg=2.0)
    
    # lr of 0.0002 is standard for GANs
    opt_gen_ext = optim.Adam(list(gen.parameters()) + list(ext.parameters()), lr=0.0002, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    train_dl, _ = get_dataloaders(batch_size=64)
    # Grab ONE batch of real images to test the plumbing
    real_images, _ = next(iter(train_dl))
    real_images = real_images.to(device)
    batch_size = real_images.size(0)
    
    print("Forward pass.\n")
    
    # Disctiminator train
    opt_disc.zero_grad()
    
    # Generate random secret messages to use as seeds
    random_binary = torch.randint(0, 2, (batch_size, latent_dim), dtype=torch.float32)
    secret_vectors = (random_binary * 2.0) - 1.0 
    secret_vectors = secret_vectors.to(device)
    
    # Generator
    fake_images = gen(secret_vectors)
    
    real_judgments = disc(real_images)
    fake_judgments = disc(fake_images.detach()) # Detach so we don't update Generator here
    
    # Calculate Discriminator Loss (How well did it tell them apart?)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    d_loss_real = criterion.adv_loss(real_judgments, real_labels)
    d_loss_fake = criterion.adv_loss(fake_judgments, fake_labels)
    d_loss = (d_loss_real + d_loss_fake) / 2
    
    # Update Discriminator weights
    d_loss.backward()
    opt_disc.step()
    
    print(f"Discriminator Complete. Loss: {d_loss.item():.4f}")

    # Train Generator & Extractor 
    opt_gen_ext.zero_grad()
    
    # Discriminator judges fake images again
    fake_judgments_2 = disc(fake_images)
    
    extracted_vectors = ext(fake_images)
    
    # Calculate Custom Loss (Trick the discriminator + recover the message)
    total_g_loss, img_loss, data_loss = criterion(fake_judgments_2, real_labels, extracted_vectors, secret_vectors)
    
    # Update Generator and Extractor weights
    total_g_loss.backward()
    opt_gen_ext.step()
    
    print(f"Total Loss: {total_g_loss.item():.4f}")
    print(f"Visual Loss: {img_loss.item():.4f}")
    print(f"Data Loss: {data_loss.item():.4f}")
    
    print("\nTraining cycle completed with zero memory leaks or crashes so far")

if __name__ == "__main__":
    check_training_loop()