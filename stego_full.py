import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

from generator import Generator
from extractor import Extractor
from mapping import MessageEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 512
    
    gen = Generator(latent_dim=latent_dim).to(device)
    ext = Extractor(latent_dim=latent_dim).to(device)
    encoder = MessageEncoder(latent_dim=latent_dim, ecc_symbols=32)
    
    try:
        gen.load_state_dict(torch.load("saved_models/rgb_blur_generator.pth", map_location=device))
        ext.load_state_dict(torch.load("saved_models/rgb_blur_extractor.pth", map_location=device))
        gen.eval()
        ext.eval()
    except FileNotFoundError:
        print("[!] Error: Could not find saved models. Ensure train.py has been run.")
        return

    secret_text = input("Enter a secret message (<=32 chars): ")
    
    try:
        binary_list = encoder.text_to_binary(secret_text)
        message_length = len(binary_list)
        secret_vector = encoder.binary_to_latent(binary_list, batch_size=1).to(device)
        original_tensor = secret_vector.squeeze().cpu()
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("Generating 64x64 RGB image with embedded payload...")
    with torch.no_grad():
        fake_image = gen(secret_vector)
        
    image_path = "secret.png"
    save_image(fake_image, image_path, normalize=True, value_range=(-1.0, 1.0))
    print(f"Image successfully saved to your folder as '{image_path}'")

    print("Loading image from disk...")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    loaded_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(loaded_image).unsqueeze(0).to(device)

    print("Extracting hidden data array...")
    with torch.no_grad():
        extracted_vector = ext(image_tensor)
        
    extracted_vector_cpu = extracted_vector.squeeze().cpu()
    extracted_binary_full = (extracted_vector_cpu > 0).float() * 2.0 - 1.0
    
    correct_bits = (extracted_binary_full[:message_length] == original_tensor[:message_length]).sum().item()
    ber = 100 - ((correct_bits / message_length) * 100)
    print(f"Bit Error Rate (BER) measured at: {ber:.2f}%")

    try:
        extracted_list = encoder.latent_to_binary(extracted_vector_cpu, message_length)
        recovered_text = encoder.binary_to_text(extracted_list)
        print(f"\nRECOVERED MESSAGE: {recovered_text}\n")
    except Exception as e:
        print(f"\nFailed to decode text. Error: {e}\n")

if __name__ == "__main__":
    main()