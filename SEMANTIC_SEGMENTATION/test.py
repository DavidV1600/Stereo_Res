import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.models.segmentation as segmentation
from dataset import val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MODEL_NAME = "deeplabv3_x2_epoch_1.pth"
MODEL_NAME = "deeplabv4_epoch_5.pth"
#MODEL_NAME = "best_model_iou0.4159.pth"
#MODEL_NAME = "deeplabv3_x2_epoch_5.pth"



def calculate_iou(preds, labels, num_classes=21):
    """
    Calculate Intersection over Union (IoU) for each class.
    """
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)  # Return the mean IoU, ignoring NaN values


def convert_bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            new_layer = nn.GroupNorm(min(32, num_channels), num_channels)
            setattr(module, name, new_layer)
        else:
            convert_bn_to_gn(child)

model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
#convert_bn_to_gn(model)  # Convert BatchNorm to GroupNorm
model = model.to(device)
model.load_state_dict(torch.load(f"./checkpoints/{MODEL_NAME}", weights_only=True))
model.eval()

criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore void class

total_loss = 0.0
total_iou = 0.0
with torch.no_grad():  # Disable gradient computation
    for idx, (images, labels) in enumerate(val_loader):
        # Move data to the GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)['out']
        loss = criterion(outputs, labels)

        # Accumulate loss
        total_loss += loss.item()

        # Calculate IoU
        preds = torch.argmax(outputs, dim=1)  # Get predicted class labels
        iou = calculate_iou(preds.cpu(), labels.cpu())
        total_iou += iou

        # Print loss and IoU every 10 batches
        if idx % 10 == 0:
            print(f"Batch {idx}, Loss: {loss.item()}, IoU: {iou}")

# Compute average loss and IoU over the entire validation set
avg_loss = total_loss / len(val_loader)
avg_iou = total_iou / len(val_loader)
print(f"Validation Loss: {avg_loss}, Mean IoU: {avg_iou}")

txtName = f"./results/{MODEL_NAME}"
with open(txtName, 'a') as fp:
    fp.write(
        f'\nAVG LOSS: {avg_loss}  - AVG IOU: {avg_iou}')