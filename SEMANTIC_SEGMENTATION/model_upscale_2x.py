import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# Adjust the path to the EDSR code
edsr_path = os.path.abspath("../EDSRPyTorch/src/model")
sys.path.append(edsr_path)

from edsr import EDSR

def load_edsr(model_path="./checkpoints/EDSR_x2.pt", scale=2, n_resblocks=32, n_feats=256, n_colors=3, res_scale=0.1):
    """
    Loads the EDSR model for 2x upscaling, assuming the model was trained
    with an internal rgb_range of 255.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EDSR expects some configuration as arguments
    class Args:
        def __init__(self):
            self.n_resblocks = n_resblocks
            self.n_feats = n_feats
            self.scale = [scale]    # Must be a list
            self.n_colors = n_colors
            self.res_scale = res_scale
            self.rgb_range = 255    # Model is trained with 255 range

    args = Args()
    model = EDSR(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def upscale_image(image, model, device):
    """
    Upscales a single image using EDSR, properly scaling
    between [0,1] <-> [0,255].
    """
    # 1) Convert PIL to tensor [0,1]
    to_tensor_01 = transforms.ToTensor()
    img_01 = to_tensor_01(image)  # shape: (3, H, W), range [0,1]

    # 2) Scale from [0,1] to [0,255]
    img_255 = img_01 * 255.0
    img_255 = img_255.unsqueeze(0).to(device)  # add batch dimension

    # 3) Inference with EDSR (expects range ~0-255)
    with torch.no_grad():
        output_255 = model(img_255)  # shape: (1, 3, H*scale, W*scale), range ~0-255

    # 4) Scale output from [0,255] back down to [0,1] and clamp
    output_01 = output_255.clamp(0, 255) / 255.0

    # 5) Convert back to PIL
    upscaled_image = transforms.ToPILImage()(output_01.squeeze(0))
    return upscaled_image

def upscale_dataset(input_dir, output_dir, model, device):
    """
    Upscales all images in a folder and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        try:
            image = Image.open(input_path).convert("RGB")
            upscaled_image = upscale_image(image, model, device)
            upscaled_image.save(output_path)
            print(f"Upscaled {image_name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

if __name__ == "__main__":
    input_dir = "./data/VOC_train/VOCdevkit/VOC2012/JPEGImages"
    output_dir = "./data/VOC_train_upscaled_2x/JPEGImages"

    model, device = load_edsr()
    upscale_dataset(input_dir, output_dir, model, device)
