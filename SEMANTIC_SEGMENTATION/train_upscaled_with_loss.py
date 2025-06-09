import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as segmentation
import os
from dataset_upscaled import train_loader, val_loader
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # For better error debugging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. First define the BatchNorm to GroupNorm conversion function
def convert_bn_to_gn(module, group_size=32):
    """Convert all BatchNorm layers to GroupNorm"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # GroupNorm with group_size channels per group (but not more than num_channels)
            num_groups = min(group_size, num_channels)
            new_layer = nn.GroupNorm(num_groups, num_channels)
            setattr(module, name, new_layer)
        else:
            convert_bn_to_gn(child, group_size)

# 2. Improved IoU calculation
def calculate_iou(preds, labels, num_classes=21, ignore_index=255):
    """Calculate mean IoU excluding ignore_index"""
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    # Exclude ignore_index from calculation
    valid_mask = labels != ignore_index
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
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

# 3. IoU/Jaccard Loss
class IoULoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        # inputs: (N, C, H, W) logits
        # targets: (N, H, W) class indices
        probs = torch.softmax(inputs, dim=1)
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Exclude ignore_index
        mask = targets != self.ignore_index
        probs = probs * mask.unsqueeze(1)
        targets_onehot = targets_onehot * mask.unsqueeze(1)
        
        intersection = (probs * targets_onehot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)  # Smoothing
        return 1 - iou.mean()

# 4. Initialize model
model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
convert_bn_to_gn(model)  # Now this is defined
model = model.to(device)

# 5. Loss functions and optimizer
ce_loss = nn.CrossEntropyLoss(ignore_index=255)
iou_loss = IoULoss(num_classes=21, ignore_index=255)
def combined_loss(outputs, targets):
    return ce_loss(outputs, targets) + iou_loss(outputs, targets)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# 6. Training loop
num_epochs = 5
ACCUMULATION_STEPS = 4
best_iou = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).long()
        
        outputs = model(images)['out']
        loss = combined_loss(outputs, labels) / ACCUMULATION_STEPS
        loss.backward()
        
        if (idx+1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * ACCUMULATION_STEPS
        
        if idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {idx}, Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")
    
    # Validation
    model.eval()
    val_metrics = {'loss': 0, 'iou': 0}
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)['out']
            
            val_metrics['loss'] += combined_loss(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            val_metrics['iou'] += calculate_iou(preds.cpu(), labels.cpu())
    
    avg_val_iou = val_metrics['iou'] / len(val_loader)
    scheduler.step(avg_val_iou)
    
    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_metrics['loss']/len(val_loader):.4f}, Val IoU: {avg_val_iou:.4f}")
    
    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        torch.save(model.state_dict(), f"checkpoints/best_model_iou{avg_val_iou:.4f}.pth")