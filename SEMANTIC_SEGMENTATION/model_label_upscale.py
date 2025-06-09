import os
import cv2
from tqdm import tqdm  # For progress bar

# Define paths
input_folder = "./data/VOC_train/VOCdevkit/VOC2012/SegmentationClass"  
output_folder = "./data/VOC_train_upscaled_4x/SegmentationClass_4x"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of label images
label_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

# Upscale each label image by 2x
for label_file in tqdm(label_files, desc="Upscaling label images"):
    # Load the label image
    label_path = os.path.join(input_folder, label_file)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # Read as-is (preserve class IDs)

    # Upscale by 2x using nearest-neighbor interpolation
    upscaled_label = cv2.resize(
        label,
        (label.shape[1] * 4, label.shape[0] * 4),  # Double the width and height
        interpolation=cv2.INTER_NEAREST,  # Nearest-neighbor interpolation
    )

    # Save the upscaled label image
    output_path = os.path.join(output_folder, label_file)
    cv2.imwrite(output_path, upscaled_label)

print(f"Upscaled {len(label_files)} label images saved to {output_folder}")