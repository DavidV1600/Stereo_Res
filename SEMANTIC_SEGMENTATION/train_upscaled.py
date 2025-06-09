import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as segmentation
from dataset_upscaled import train_loader #, val_loader
from dataset import val_loader
import numpy as np

# Enable this for debugging CUDA errors
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define IoU calculation function
def calculate_iou(preds, labels, num_classes=21):
    """Calculate Intersection over Union (IoU) for each class."""
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# Load and modify the model
model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

# Convert BatchNorm to GroupNorm to handle batch_size=1
def convert_bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            new_layer = nn.GroupNorm(min(32, num_channels), num_channels)
            setattr(module, name, new_layer)
        else:
            convert_bn_to_gn(child)

def freeze_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()  # Freeze running mean/var
        module.requires_grad_(False)  # Freeze parameters

convert_bn_to_gn(model)

#model.apply(freeze_bn)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)  # VOC has 21 classes (0-20) + 255 for ignore
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
ACCUMULATION_STEPS = 4

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0.0
    
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()  # Ensure labels are long integers
        
        # Verify label values are valid
        assert labels.min() >= 0 and labels.max() < 21 or labels.max() == 255, \
               f"Invalid label values: min={labels.min()}, max={labels.max()}"
        
        # Forward pass
        outputs = model(images)['out']
        
        # Check output and label shapes
        assert outputs.dim() == 4 and labels.dim() == 3, \
               f"Shapes mismatch: outputs {outputs.shape}, labels {labels.shape}"
        
        loss = criterion(outputs, labels)
        #loss = criterion(outputs, labels) / ACCUMULATION_STEPS
        loss.backward()
        
        epoch_loss += loss.item()
        #epoch_loss += loss.item() * ACCUMULATION_STEPS
        

        optimizer.step()
        optimizer.zero_grad()
        # if (idx + 1) % ACCUMULATION_STEPS == 0 or (idx + 1) == len(train_loader):
        #     optimizer.step()
        #     optimizer.zero_grad()
        
        if idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{idx}/{len(train_loader)}], Loss: {loss.item() * ACCUMULATION_STEPS}")

    scheduler.step()
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss}")
    
    # Validation phase
    # model.eval()
    # val_loss = 0.0
    # val_iou = 0.0
    
    # with torch.no_grad():
    #     for idx, (images, labels) in enumerate(val_loader):
    #         images = images.to(device)
    #         labels = labels.to(device).long()
            
    #         outputs = model(images)['out']
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()
            
    #         preds = torch.argmax(outputs, dim=1)
    #         iou = calculate_iou(preds.cpu(), labels.cpu())
    #         val_iou += iou
            
    #         if idx % 10 == 0:
    #             print(f"Val Batch {idx}, Loss: {loss.item()}, IoU: {iou}")

    # avg_val_loss = val_loss / len(val_loader)
    # avg_val_iou = val_iou / len(val_loader)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss}, Mean IoU: {avg_val_iou}")
    
    torch.save(model.state_dict(), f"./checkpoints/deeplabv3_x2_epoch_{epoch+1}.pth")