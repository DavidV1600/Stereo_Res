import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# Add path for the SRFBN model
srfbn_path = os.path.abspath("../MHCA/edsr")
sys.path.append(srfbn_path)

# Import the SRFBN model
from model import Model

def load_edsr_model(model_path="./checkpoints/model_single_input_IXI_x2.pth", scale=2, n_colors=3, n_feats=64, n_resblocks=16, res_scale=0.1, shift_mean=False, use_mhca_2=False, use_mhca_3=False):
    """Loads the EDSR model with optional MHCA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the arguments for the model initialization
    args = type('', (), {})()  # Create a simple object for passing arguments
    args.scale = [scale]  # Scale factor for the model
    args.n_colors = n_colors  # Number of channels (RGB images: 3 channels)
    args.n_feats = n_feats  # Number of features in the model
    args.n_resblocks = n_resblocks  # Number of residual blocks
    args.res_scale = res_scale  # Residual scaling factor
    args.shift_mean = shift_mean  # Whether to apply mean shift (usually false in SR models)
    args.use_mhca_2 = use_mhca_2  # Flag for the second MHCA module
    args.use_mhca_3 = use_mhca_3  # Flag for the third MHCA module
    args.rgb_range = 255  # Assuming RGB range 0-255 for standard images

    # Initialize the EDSR model
    model = Model(args)

    # Load the model weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove 'module.' prefix from the state_dict keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = v

    # Load the state_dict into the model
    model.load_state_dict(new_state_dict)

    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model, device

def upscale_image(image, model, device, scale_factor=2):
    """Upscales a single image using EDSR."""
    transform = transforms.ToTensor()  # Convert PIL Image to tensor
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)  # Get the model's output

    upscaled_image = output.squeeze(0)  # Remove batch dimension

    # Convert tensor back to PIL Image
    upscaled_image = transforms.ToPILImage()(upscaled_image)
    return upscaled_image

def upscale_dataset(input_dir, output_dir, model, device, scale_factor=2):
    """Upscales all images in a folder and saves them."""
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            upscaled_image = upscale_image(image, model, device, scale_factor)
            upscaled_image.save(output_path)
            print(f"Upscaled {image_name} and saved to {output_path}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

if __name__ == "__main__":
    # Paths
    input_dir = "./data/VOC_train/VOCdevkit/VOC2012/JPEGImages"
    output_dir = "./data/VOC_train_upscaled_2x/JPEGImages"

    # Load model
    model, device = load_edsr_model(model_path="./checkpoints/EDSR-x2.pth", scale=2, n_feats=64, n_resblocks=16)

    # Upscale images
    upscale_dataset(input_dir, output_dir, model, device, scale_factor=2)
