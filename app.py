import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from generator import Generator
from extractor import Extractor
from mapping import MessageEncoder

# @st.cache_resource loads the AI brains into memory once for speed
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 512
    
    gen = Generator(latent_dim=latent_dim).to(device)
    ext = Extractor(latent_dim=latent_dim).to(device)
    encoder = MessageEncoder(latent_dim=latent_dim, ecc_symbols=32)
    
    try:
        gen.load_state_dict(torch.load("saved_models/rgb_blur_generator.pth", map_location=device, weights_only=True))
        ext.load_state_dict(torch.load("saved_models/rgb_blur_extractor.pth", map_location=device, weights_only=True))
        gen.eval()
        ext.eval()
        return gen, ext, encoder, device, latent_dim, True
    except FileNotFoundError:
        return None, None, None, device, latent_dim, False

gen, ext, encoder, device, latent_dim, models_loaded = load_models()

st.set_page_config(page_title="StegoAI", layout="centered")
st.title("StegoAI")
st.markdown("Hide encrypted text payloads inside AI-generated color images.")

if not models_loaded:
    st.error("Could not find trained models! Make sure 'generator.pth' and 'extractor.pth' are in the 'saved_models' folder.")
    st.stop()

tab1, tab2 = st.tabs(["🔒 Encode Message", "🔓 Decode Image"])

# TAB 1: ENCODE 
with tab1:
    st.subheader("Generate an Image with a Secret Payload")
    secret_text = st.text_input("Enter a secret message (max ~32 chars):")
    
    if st.button("Generate Image"):
        if secret_text:
            try:
                binary_list = encoder.text_to_binary(secret_text)
                secret_vector = encoder.binary_to_latent(binary_list, batch_size=1).to(device)
                
                with torch.no_grad():
                    fake_image = gen(secret_vector)
                
                img_shifted = (fake_image + 1.0) / 2.0
                img_shifted = torch.clamp(img_shifted, 0, 1)
                img_pil = transforms.ToPILImage()(img_shifted.squeeze(0))
                
                st.image(img_pil, caption="AI Generated 64x64 Image", width=256)
                
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download secret_rgb.png",
                    data=byte_im,
                    file_name="secret_rgb.png",
                    mime="image/png"
                )
                st.success("Image generated successfully. You can now download it.")
                
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            st.warning("Enter a message first")

# TAB 2: DECODE 
with tab2:
    st.subheader("Extract a Hidden Payload from an Image")
    uploaded_file = st.file_uploader("Upload a StegoAI-generated PNG:", type=["png"])
    
    if st.button("Decode Message"):
        if uploaded_file is not None:
            loaded_image = Image.open(uploaded_file).convert("RGB") 
            
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_tensor = transform(loaded_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                extracted_vector = ext(image_tensor)
            extracted_vector_cpu = extracted_vector.squeeze().cpu()
            
            recovered_text = None
            with st.spinner('Decrypting payload...'):
                for length in range(8, latent_dim + 1, 8):
                    try:
                        extracted_list = encoder.latent_to_binary(extracted_vector_cpu, length)
                        text_guess = encoder.binary_to_text(extracted_list)
                        
                        if len(text_guess) > 0:
                            recovered_text = text_guess
                            break
                    except Exception:
                        continue
            
            if recovered_text:
                st.success(f"**DECODED MESSAGE:** {recovered_text}")
                st.balloons() 
            else:
                st.error("Failed to decode text. The message may be too corrupted or the image contains no data.")
        else:
            st.warning("Please upload an image first!")