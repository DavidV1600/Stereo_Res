import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/home/david/PycharmProjects/SSRDEF-Net/SEMANTIC_SEGMENTATION/Fast-SCNN-pytorch')
from models.fast_scnn import FastSCNN
from ade20k_dataset import train_loader  # or train_loader_sr
import numpy as np
from PIL import Image
import torchvision
print(torchvision.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 150  # ADE20K has 150 classes

def calculate_iou(preds, labels, num_classes=NUM_CLASSES):
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

def mask_transform(mask):
    mask = mask.resize((520, 520), Image.NEAREST)
    mask = np.array(mask)
    mask[mask == 0] = 255      # Set ignore
    mask = mask - 1            # 1-150 -> 0-149, 255->254
    mask[mask == 254] = 255    # Set ignore back to 255
    mask = torch.tensor(mask, dtype=torch.long)
    return mask

model = FastSCNN(num_classes=150)
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{idx}/{len(train_loader)}], Loss: {loss.item()}")
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss}")
    torch.save(model.state_dict(), f"./checkpoints/fastscnn_ade20k_epoch_{epoch+1}.pth")

    # Sanity check in your dataloader
    if idx == 0:
        print("Unique mask values:", torch.unique(labels))
