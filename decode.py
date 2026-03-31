import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Import your custom modules
from extractor import Extractor
from mapping import MessageEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    image_path = "secret.png"
    
    print("--- STANDALONE DECODER ---")
    
    # 1. Initialize Extractor and Encoder
    ext = Extractor(latent_dim=latent_dim).to(device)
    encoder = MessageEncoder(latent_dim=latent_dim, ecc_symbols=16)
    
    # 2. Load the Extractor's trained brain
    try:
        ext.load_state_dict(torch.load("saved_models/extractor.pth", map_location=device, weights_only=True))
        ext.eval()
        print("[+] Extractor model loaded successfully.")
    except FileNotFoundError:
        print("[!] Error: Could not find 'extractor.pth'.")
        return

    # 3. Load the Image
    if not os.path.exists(image_path):
        print(f"[!] Error: Could not find '{image_path}'. Run stego_full.py first to generate it.")
        return
        
    print(f"[+] Loading '{image_path}' from disk...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    loaded_image = Image.open(image_path)
    image_tensor = transform(loaded_image).unsqueeze(0).to(device)

    # 4. Extract the hidden data
    print("[+] Extracting data from pixels...")
    with torch.no_grad():
        extracted_vector = ext(image_tensor)
        
    extracted_vector_cpu = extracted_vector.squeeze().cpu()
    
    # 5. Blind Decoding (Brute-forcing the message length)
    # We don't know the original text length, so we try decoding chunks of 8 bits (1 byte) 
    # at a time until Reed-Solomon confirms the data is valid!
    print("[+] Attempting Reed-Solomon reconstruction...\n")
    
    recovered_text = None
    
    # Try lengths from 1 byte (8 bits) up to max capacity (256 bits)
    for length in range(8, latent_dim + 1, 8):
        try:
            # Grab 'length' amount of bits
            extracted_list = encoder.latent_to_binary(extracted_vector_cpu, length)
            # Attempt to decode
            text_guess = encoder.binary_to_text(extracted_list)
            
            # CRITICAL FIX: Only break if it decoded ACTUAL text, not just 0-byte ECC padding
            if len(text_guess) > 0:
                recovered_text = text_guess
                break 
                
        except Exception:
            # RS decoding failed (wrong length or too much corruption), try the next size
            continue

    if recovered_text:
        print("====================================")
        print(f"DECODED MESSAGE: {recovered_text}")
        print("====================================")
    else:
        print("Failed to decode text. The message may be too corrupted, or the image contains no hidden data.")

if __name__ == "__main__":
    main()