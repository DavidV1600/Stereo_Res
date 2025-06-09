import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation

# Define label transformations
def label_transform(label):
    # Resize the label to 520x520
    label = label.resize((1520, 1520), Image.NEAREST)  # Use nearest-neighbor interpolation
    
    # Convert PIL image to NumPy array
    label = np.array(label)
    
    # Convert NumPy array to tensor
    label = torch.tensor(label, dtype=torch.long)  # Convert to a tensor of type long
    return label

class CustomVOCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        # Get all image filenames (assuming .jpg images and .png masks)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.mask_files = [f.replace('.jpg', '.png') for f in self.image_files]
        
        # Verify that masks exist for all images
        self.valid_pairs = []
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_file, mask_file))
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.valid_pairs[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = Image.open(mask_path)
        
        # Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Final safety check - ensure no values above 20 except 255
        invalid_indices = (mask > 20) & (mask != 255)
        if invalid_indices.any():
            mask[invalid_indices] = 0
        
        return image, mask

# Same transforms as the first pipeline
image_transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Add this function to preprocess and visualize masks
def check_and_fix_masks():
    # Load a few masks to check their values
    sample_dir = './data/VOC_train_upscaled_2x/SegmentationClass_2x'
    files = os.listdir(sample_dir)[:5]  # Check first 5 files
    
    for file in files:
        mask_path = os.path.join(sample_dir, file)
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        
        print(f"File: {file}")
        print(f"Shape: {mask_np.shape}")
        print(f"Unique values: {np.unique(mask_np)}")
        print("---")

# Update the mask_transform function to properly handle the invalid values
def mask_transform(mask):
    # Resize the label to 520x520
    mask = mask.resize((520, 520), Image.NEAREST)
    
    # Convert to numpy array
    mask_np = np.array(mask)
    
    # If mask is multi-channel, convert to single channel
    if len(mask_np.shape) > 2:
        mask_np = mask_np[:, :, 0]
    
    # Map values that are greater than 20 (except 255) to valid range
    # Create a new array for the mapped values
    mapped_mask = np.zeros_like(mask_np)
    
    # Keep 0-20 as is, map other values (except 255) to 0
    for value in np.unique(mask_np):
        if 0 <= value <= 20 or value == 255:
            # Keep valid class indices as they are
            mapped_mask[mask_np == value] = value
        else:
            # Map invalid values to background class (0)
            mapped_mask[mask_np == value] = 0
    
    # Convert to tensor
    mapped_mask = torch.tensor(mapped_mask, dtype=torch.long)
    return mapped_mask

# Create datasets (now matching the first pipeline's logic)
train_dataset = CustomVOCDataset(
    image_dir='./data/VOC_train_upscaled_2x/JPEGImages',
    mask_dir='./data/VOC_train_upscaled_2x/SegmentationClass_2x',
    image_transform=image_transform,
    mask_transform=mask_transform
)

# val_dataset = CustomVOCDataset(
#     image_dir='./data/VOC_train_upscaled_2x/JPEGImages',
#     mask_dir='./data/VOC_train_upscaled_2x/SegmentationClass_2x',
#     image_transform=image_transform,
#     mask_transform=mask_transform
# )

# Create dataloaders (same batch size as the first pipeline)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Verification
def verify_dataloader(loader):
    for images, masks in loader:
        print("Image shape:", images.shape)  # Should be [4, 3, 520, 520]
        print("Mask shape:", masks.shape)    # Should be [4, 520, 520]
        print("Unique mask values:", torch.unique(masks))  # Should be 0-20 + 255
        break

print("Training samples:")
verify_dataloader(train_loader)
# print("\nValidation samples:")
# verify_dataloader(val_loader)