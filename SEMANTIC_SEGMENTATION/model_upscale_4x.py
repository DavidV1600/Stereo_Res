import os
import torch
import numpy as np
from PIL import Image  # ✅ Import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def load_realesrgan(model_name="RealESRGAN_x4plus"):
    """Loads the pretrained Real-ESRGAN model."""
    model_path = "./checkpoints/RealESRGAN_x4plus.pth"
    
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RealESRGANer(
        model_path=model_path,
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32),
        scale=4,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        device=device  # ✅ Correct way to set device
    )
    
    return model

def upscale_image(image, model, scale_factor=4):
    """Upscales a single image using Real-ESRGAN."""
    image_np = np.array(image)  # ✅ Convert PIL image to NumPy array
    upscaled_image, _ = model.enhance(image_np, outscale=scale_factor)  # ✅ Pass NumPy array
    return Image.fromarray(upscaled_image)  # ✅ Convert back to PIL Image

def upscale_dataset(input_dir, output_dir, model, scale_factor=4):
    """Upscales all images in a folder and saves them."""
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            upscaled_image = upscale_image(image, model, scale_factor)
            upscaled_image.save(output_path)
            print(f"Upscaled {image_name} and saved to {output_path}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

if __name__ == "__main__":
    # Paths
    input_dir = "./data/VOC_train/VOCdevkit/VOC2012/JPEGImages"
    output_dir = "./data/VOC_train_upscaled_4x/JPEGImages"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_realesrgan()
    # Upscale images
    upscale_dataset(input_dir, output_dir, model)
