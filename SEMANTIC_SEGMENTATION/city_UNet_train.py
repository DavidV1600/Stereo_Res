import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import os
import random
from torch.cuda.amp import autocast, GradScaler

# ----------------------------
# 1. CONFIGURATION
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_CLASSES = 19  # For Cityscapes

# Cityscapes-specific settings
# The Cityscapes dataset has 34 classes, but only 19 are used for training/evaluation
# We need to map the 34 classes to the 19 evaluation classes or to 255 (ignore)
CITYSCAPES_LABEL_MAP = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
    14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

# Paths (modify as needed)
ORIGINAL_DATA_DIR = "/home/david/PycharmProjects/SSRDEF-Net/SEMANTIC_SEGMENTATION/data/city_scapes"

# ----------------------------
# 2. DATASET CLASSES WITH LABEL REMAPPING
# ----------------------------
class CityscapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, split='train', transform=None, target_transform=None, crop_size=None):
        self.img_dir = os.path.join(img_dir, 'leftImg8bit', split)
        self.mask_dir = os.path.join(mask_dir, 'gtFine', split)
        self.transform = transform
        self.target_transform = target_transform
        self.crop_size = crop_size
        self.files = self._get_file_pairs()
        
    def _get_file_pairs(self):
        pairs = []
        for city in os.listdir(self.img_dir):
            city_img_dir = os.path.join(self.img_dir, city)
            city_mask_dir = os.path.join(self.mask_dir, city)
            for img_name in os.listdir(city_img_dir):
                if img_name.endswith('_leftImg8bit.png'):
                    mask_name = img_name.replace('leftImg8bit', 'gtFine_labelIds')
                    pairs.append((
                        os.path.join(city_img_dir, img_name),
                        os.path.join(city_mask_dir, mask_name)
                    ))
        return pairs
    
    def __len__(self):
        return len(self.files)
    
    def _remap_labels(self, mask):
        # Convert mask to numpy array if it's not already
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Create a new mask with remapped labels
        new_mask = np.ones_like(mask) * 255  # Initialize with ignore label
        
        # Remap using the dictionary
        for old_label, new_label in CITYSCAPES_LABEL_MAP.items():
            new_mask[mask == old_label] = new_label
            
        return new_mask
    
    def _random_crop(self, image, mask):
        # Get current dimensions
        w, h = image.size
        
        # Determine crop size
        crop_h, crop_w = self.crop_size
        
        # Get random crop coordinates
        if w > crop_w and h > crop_h:
            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
            
            # Crop images
            image = image.crop((x, y, x + crop_w, y + crop_h))
            mask = mask.crop((x, y, x + crop_w, y + crop_h))
            
        return image, mask
    
    def __getitem__(self, idx):
        img_path, mask_path = self.files[idx]
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Random crop if specified
        if self.crop_size:
            image, mask = self._random_crop(image, mask)
        
        # Apply image transformations
        if self.transform:
            image = self.transform(image)
        
        # Remap labels and convert mask to tensor
        mask = self._remap_labels(mask)
        
        # Apply any mask transformations
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()
            
        return image, mask

# ----------------------------
# 3. MODEL ARCHITECTURE (UNet)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.up1 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        
        # Decoder
        x = self.upsample(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x)
        x = self.final(x)
        return x

# ----------------------------
# 4. TRAINING UTILITIES WITH MEMORY OPTIMIZATIONS
# ----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_miou = 0.0
    history = {'train_loss': [], 'val_miou': []}
    scaler = GradScaler()  # For mixed precision training
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # Training phase
        for i, (images, masks) in enumerate(train_loader):
            # Check if masks have any valid pixels (not 255)
            if (masks == 255).all():
                print(f"WARNING: Batch {i} has all ignore labels, skipping.")
                continue
                
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use mixed precision for forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Print batch info
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
                # Print mask stats for debugging
                valid_pixels = (masks != 255).sum().item()
                total_pixels = masks.numel()
                print(f"  Valid pixels: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
                print(f"  Unique labels in batch: {torch.unique(masks)}")
            
            # Free memory
            torch.cuda.empty_cache()
        
        # Validation phase
        val_miou = evaluate(model, val_loader)
        
        # Record metrics
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_miou'].append(val_miou)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Loss: {epoch_loss/len(train_loader):.4f} | '
              f'Val mIoU: {val_miou:.4f} | '
              f'Time: {epoch_time:.1f}s')
    
    return history

def evaluate(model, dataloader):
    model.eval()
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.cpu().numpy()
            
            # Use mixed precision inference
            with autocast():
                outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Only consider valid pixels (not 255)
            for pred, mask in zip(preds, masks):
                valid = mask != 255
                if valid.any():  # Only process if there are valid pixels
                    conf_matrix += confusion_matrix(
                        mask[valid].flatten(),
                        pred[valid].flatten(),
                        labels=list(range(NUM_CLASSES))
                    )
            
            # Free memory
            torch.cuda.empty_cache()
    
    # Calculate IoU per class
    iou_per_class = []
    for i in range(NUM_CLASSES):
        tp = conf_matrix[i,i]
        fp = conf_matrix[:,i].sum() - tp
        fn = conf_matrix[i,:].sum() - tp
        iou = tp / (tp + fp + fn + 1e-10)
        iou_per_class.append(iou)
    
    return np.mean(iou_per_class)

# ----------------------------
# 5. MAIN EXPERIMENT WITH CROPPING AND MEMORY OPTIMIZATIONS
# ----------------------------
def run_experiment():
    # Define crop size to reduce memory usage
    CROP_SIZE = (512, 512)  # (height, width)
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("=== Preparing Datasets ===")
    # Dataset with random cropping for training
    orig_train = CityscapesDataset(
        ORIGINAL_DATA_DIR, 
        ORIGINAL_DATA_DIR, 
        'train', 
        transform=transform,
        crop_size=CROP_SIZE
    )
    
    # For validation, we'll use the whole image
    orig_val = CityscapesDataset(
        ORIGINAL_DATA_DIR, 
        ORIGINAL_DATA_DIR, 
        'val', 
        transform=transform
    )
    
    print(f"Training set size: {len(orig_train)}")
    print(f"Validation set size: {len(orig_val)}")
    
    # Verify label distribution
    print("Checking label distribution in first few samples...")
    for i in range(min(3, len(orig_train))):
        _, mask = orig_train[i]
        unique_labels = torch.unique(mask)
        print(f"Sample {i} unique labels: {unique_labels}")
        valid_ratio = (mask != 255).float().mean().item() * 100
        print(f"  Valid pixel ratio: {valid_ratio:.1f}%")
    
    # Create dataloaders
    orig_train_loader = DataLoader(
        orig_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True
    )
    
    orig_val_loader = DataLoader(
        orig_val, 
        batch_size=BATCH_SIZE,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True
    )
    
    print("=== Training on Cropped Images ===")
    model_orig = UNet().to(DEVICE)
    optimizer = optim.Adam(model_orig.parameters(), lr=1e-4)
    
    # Use CrossEntropyLoss with ignore_index=255
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # First, try to run a single batch to validate everything works
    print("Testing with a single batch...")
    try:
        for images, masks in orig_train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            with autocast():
                outputs = model_orig(images)
                loss = criterion(outputs, masks)
                
            print(f"Single batch test successful! Loss: {loss.item():.4f}")
            break
    except Exception as e:
        print(f"Error in single batch test: {e}")
        # More detailed error information
        import traceback
        traceback.print_exc()
        return
    
    # If successful, run the full training
    orig_history = train_model(
        model_orig, 
        orig_train_loader, 
        orig_val_loader, 
        criterion, 
        optimizer, 
        NUM_EPOCHS
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    orig_miou = evaluate(model_orig, orig_val_loader)
    print(f"Original Resolution mIoU: {orig_miou:.4f}")

if __name__ == "__main__":
    run_experiment()