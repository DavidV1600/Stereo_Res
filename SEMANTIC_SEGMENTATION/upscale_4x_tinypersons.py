import os
import torch
import shutil
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def load_realesrgan(model_name="RealESRGAN_x4plus"):
    model_path = "./checkpoints/RealESRGAN_x4plus.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RealESRGANer(
        model_path=model_path,
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32),
        scale=4,
        tile=512,  # Process in 512x512 tiles
        tile_pad=32,
        pre_pad=0,
        device=device
    )
    
    return model

def upscale_yolo_dataset(base_path, output_path, scale=4):
    """Upscale a YOLO dataset, preserving the labels"""
    # Load model
    model = load_realesrgan()
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        # Create image directories
        os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
        
        # Copy label directories exactly as they are (labels don't need modification)
        src_labels = os.path.join(base_path, 'labels', split)
        dst_labels = os.path.join(output_path, 'labels', split)
        
        if os.path.exists(src_labels) and not os.path.exists(dst_labels):
            shutil.copytree(src_labels, dst_labels)
        
        # Upscale images
        img_dir = os.path.join(base_path, 'images', split)
        out_img_dir = os.path.join(output_path, 'images', split)
        
        if not os.path.exists(img_dir):
            continue
            
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(img_dir, img_name)
            out_path = os.path.join(out_img_dir, img_name)
            
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Upscale image
                img_np = np.array(img)
                upscaled_img, _ = model.enhance(img_np, outscale=scale)
                
                # Save upscaled image
                Image.fromarray(upscaled_img).save(out_path)
                print(f"Upscaled {split}/{img_name}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Copy dataset.yaml with updated path
    with open(os.path.join(base_path, 'dataset.yaml'), 'r') as f:
        yaml_content = f.read()
    
    with open(os.path.join(output_path, 'dataset.yaml'), 'w') as f:
        # Update the path in the yaml
        yaml_content = yaml_content.replace(os.path.abspath(base_path), os.path.abspath(output_path))
        f.write(yaml_content)
    
    print(f"Dataset upscaled and saved to {output_path}")

if __name__ == "__main__":
    upscale_yolo_dataset(
        base_path="./data2/output",          # Original YOLO dataset
        output_path="./data2/output_sr_4x",  # SR output path
        scale=4                              # Upscaling factor
    )
