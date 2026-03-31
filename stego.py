import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

# Import your custom modules
from generator import Generator
from extractor import Extractor
from mapping import MessageEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    
    gen = Generator(latent_dim=latent_dim).to(device)
    ext = Extractor(latent_dim=latent_dim).to(device)
    encoder = MessageEncoder(latent_dim=latent_dim, ecc_symbols=16)
    
    try:
        gen.load_state_dict(torch.load("saved_models/generator.pth", map_location=device))
        ext.load_state_dict(torch.load("saved_models/extractor.pth", map_location=device))
        gen.eval()
        ext.eval()
    except FileNotFoundError:
        print("[!] Error: Could not find saved models. Ensure train.py has been run.")
        return

    secret_text = input("Enter a short secret message: ")
    
    try:
        # Get the binary data from your mapping script
        binary_list = encoder.text_to_binary(secret_text)
        message_length = len(binary_list)
        
        # Convert 0/1 to -1.0/1.0 and shape it for the Generator (1 batch, 256 bits)
        secret_vector = encoder.binary_to_latent(binary_list, batch_size=1).to(device)

        original_tensor = secret_vector.clone().cpu().squeeze()
    except Exception as e:
        print(f"Encoding error: {e}")
        return
    
    print("Generating AI image with embedded payload...")
    with torch.no_grad():
        fake_image = gen(secret_vector) # Make sure secret_vector matches your variable name
        
    print("\n--- RAM TEST (Direct from Memory) ---")
    with torch.no_grad():
        ram_vector = ext(fake_image)
    ram_vector_cpu = ram_vector.squeeze().cpu()
    ram_binary_full = (ram_vector_cpu > 0).float() * 2.0 - 1.0
    ram_correct = (ram_binary_full[:message_length] == original_tensor[:message_length]).sum().item()
    ram_ber = 100 - ((ram_correct / message_length) * 100)
    print(f"RAM Bit Error Rate (BER): {ram_ber:.2f}%")

    print("\nSaving image to disk...")
    image_path = "secret.png"
    save_image(fake_image, image_path, normalize=True, value_range=(-1.0, 1.0))
    print(f"Image successfully saved to your folder as '{image_path}'")

    print("\n--- DISK TEST (From PNG File) ---")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    loaded_image = Image.open(image_path)
    image_tensor = transform(loaded_image).unsqueeze(0).to(device)

    with torch.no_grad():
        extracted_vector = ext(image_tensor)
        
    extracted_vector_cpu = extracted_vector.squeeze().cpu()
    extracted_binary_full = (extracted_vector_cpu > 0).float() * 2.0 - 1.0
    disk_correct = (extracted_binary_full[:message_length] == original_tensor[:message_length]).sum().item()
    disk_ber = 100 - ((disk_correct / message_length) * 100)
    print(f"Disk Bit Error Rate (BER): {disk_ber:.2f}%")

    '''
    print("Generating AI image with embedded payload...")
    with torch.no_grad():
        fake_image = gen(secret_vector)
    
    
    image_path = "secret.png"
    save_image(fake_image, image_path, normalize=True, value_range=(-1.0, 1.0)) #added val range
    print(f"Image successfully saved to your folder as '{image_path}'")

    print("Loading image from disk...")
    # We must transform the loaded image exactly how the AI expects it (grayscale, normalized)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    loaded_image = Image.open(image_path)
    image_tensor = transform(loaded_image).unsqueeze(0).to(device)

    print("Extracting hidden data array...")
    with torch.no_grad():
        extracted_vector = ext(image_tensor)
        
    # Convert extracted -1.0/1.0 back to 0/1
    extracted_vector_cpu = extracted_vector.squeeze().cpu()
    
    # Measure the Bit Error Rate (BER)
    extracted_binary_full = (extracted_vector_cpu > 0).float() * 2.0 - 1.0
    correct_bits = (extracted_binary_full[:message_length] == original_tensor[:message_length]).sum().item()
    ber = 100 - ((correct_bits / message_length) * 100)
    print(f"Bit Error Rate (BER) measured at: {ber:.2f}%")

    try:
        # Decode the binary back to text using your mapping script
        extracted_list = encoder.latent_to_binary(extracted_vector_cpu, message_length)
        recovered_text = encoder.binary_to_text(extracted_list)
        
        print(f"RECOVERED MESSAGE: {recovered_text}")
    except Exception as e:
        print(f"\nFailed to decode text. The BER might be too high for ECC to fix. Error: {e}")
'''

if __name__ == "__main__":
    main()