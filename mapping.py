import torch
from reedsolo import RSCodec, ReedSolomonError

class MessageEncoder:
    def __init__(self, latent_dim=100, ecc_symbols=10): 
        self.latent_dim = latent_dim #latent_dim: size of the noise vector the GAN expects (usually 100).
        self.ecc_symbols = ecc_symbols
        self.rs = RSCodec(self.ecc_symbols) 

    def text_to_binary(self, text):
        # Ascii to Binary conversion

        raw_bytes = text.encode('utf-8')
        encoded_bytes = self.rs.encode(raw_bytes)

        binary_list = []
        for byte in encoded_bytes:
            bin_str = format(byte, '08b')
            for bit in bin_str:
                binary_list.append(int(bit))
                
        return binary_list

    def binary_to_latent(self, binary_list, batch_size=1):
        #Embeds the binary list into a PyTorch latent vector (tensor).
        message_length = len(binary_list)
        
        if message_length > self.latent_dim:
            raise ValueError(f"Message+ECC too long. Max bits: {self.latent_dim}, Your bits: {message_length}")

        latent_vectors = torch.randn(batch_size, self.latent_dim)
        
        message_tensor = torch.tensor(binary_list, dtype=torch.float32)
        message_tensor = (message_tensor * 2.0) - 1.0 

        for i in range(batch_size):
            latent_vectors[i, :message_length] = message_tensor

        return latent_vectors

if __name__ == "__main__":
    encoder = MessageEncoder(latent_dim=256, ecc_symbols=10)
    secret_message = input("Enter your secret message: ")

    try:
        # Original size without ECC (1 byte per char)
        raw_bits = len(secret_message) * 8
        
        # New size with ECC
        binary_data = encoder.text_to_binary(secret_message)
        ecc_bits = len(binary_data)
        
        print(f"\nOriginal Text: '{secret_message}'")
        print(f"Raw Message Size: {raw_bits} bits")
        print(f"Protected Size (Message + Error Correction): {ecc_bits} bits")
        
        # Verify it fits in the vector
        latent_batch = encoder.binary_to_latent(binary_data, batch_size=1)
        print(f"\nSuccessfully embedded into Latent Vector of size: {latent_batch.shape}")
        print(f"Space remaining in vector: {encoder.latent_dim - ecc_bits} bits")
        
    except ValueError as e:
        print(f"\n[!] ERROR: {e}")