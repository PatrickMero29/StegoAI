import torch
from generator import Generator
from extractor import Extractor

def measure_bit_error_rate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    batch_size = 64
    
    gen = Generator(latent_dim=latent_dim).to(device)
    ext = Extractor(latent_dim=latent_dim).to(device)
    
    # Load the trained weights
    try:
        gen.load_state_dict(torch.load("saved_models/generator.pth", map_location=device))
        ext.load_state_dict(torch.load("saved_models/extractor.pth", map_location=device))
        print("Successfully loaded trained models\n")
    except FileNotFoundError:
        print("Error")
        return

    # Ecaluation mode: turns off training-specific layers like dropout
    gen.eval()
    ext.eval()

    random_binary = torch.randint(0, 2, (batch_size, latent_dim), dtype=torch.float32)
    original_messages = (random_binary * 2.0) - 1.0 
    original_messages = original_messages.to(device)

    with torch.no_grad():
        fake_images = gen(original_messages)
        
        extracted_messages = ext(fake_images)

    # Any output > 0 is guessed as 1.0, and < 0 is -1.0
    extracted_binary = (extracted_messages > 0).float() * 2.0 - 1.0
    
    # Compare the extracted bits directly to the original bits
    correct_bits = (extracted_binary == original_messages).sum().item()
    total_bits = batch_size * latent_dim
    
    accuracy = (correct_bits / total_bits) * 100
    ber = 100 - accuracy
    
    print(f"Total Bits Tested: {total_bits}")
    print(f"Correctly Recovered: {correct_bits}")
    print(f"Extraction Accuracy: {accuracy:.2f}%")
    print(f"Bit Error Rate (BER): {ber:.2f}%\n")
    
    if accuracy == 100.0:
        print("Message intact, no errors")
    elif accuracy > 90.0:
        print("Error Correction (ECC) can fix the rest")
    else:
        print("Increase lambda_msg in training")

if __name__ == "__main__":
    measure_bit_error_rate()