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
    
    def latent_to_binary(self, latent_vector, message_length):
            extracted_bits = []
            for i in range(message_length):
                if latent_vector[i] > 0:
                    extracted_bits.append(1)
                else:
                    extracted_bits.append(0)
            return extracted_bits

    def binary_to_text(self, binary_list):
        byte_array = bytearray()
        for i in range(0, len(binary_list), 8):
            byte_chunk = binary_list[i:i+8]
            byte_str = "".join(str(b) for b in byte_chunk) 
            # Convert string to integer base 2
            byte_array.append(int(byte_str, 2))
            
        # rs.decode returns a tuple: (decoded_message, ecc_bytes, error_locations)
        decoded_bytes, _, _ = self.rs.decode(byte_array)
        
        return decoded_bytes.decode('utf-8')

if __name__ == "__main__":
    encoder = MessageEncoder(latent_dim=256, ecc_symbols=10)
    secret_message = input("Enter your secret message: ")
    
    try:
        binary_data = encoder.text_to_binary(secret_message)
        ecc_bits = len(binary_data)
        
        latent_batch = encoder.binary_to_latent(binary_data, batch_size=1)
        
        print(f"\n[ENCODE] Protected Size: {ecc_bits} bits")
        print(f"[ENCODE] Embedded into Vector of shape: {latent_batch.shape}")
        
        first_vector = latent_batch[0] 
        
        recovered_bits = encoder.latent_to_binary(first_vector, ecc_bits)
        recovered_text = encoder.binary_to_text(recovered_bits)
        
        print(f"\n[DECODE] Extracted Text: '{recovered_text}'")
        
        if secret_message == recovered_text:
            print("\nSUCCESS")
            
    except Exception as e:
        print(f"\n[!] ERROR: {e}")