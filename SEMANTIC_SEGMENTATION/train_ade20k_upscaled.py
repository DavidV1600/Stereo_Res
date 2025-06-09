import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as segmentation
from ade20k_dataset import train_loader_sr  # Uses the upscaled ADE20K dataloader
import numpy as np

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

#model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT, num_classes=NUM_CLASSES)
model = segmentation.deeplabv3_resnet18(weights=None, num_classes=150)

model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for idx, (images, labels) in enumerate(train_loader_sr):
        images = images.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{idx}/{len(train_loader_sr)}], Loss: {loss.item()}")
    scheduler.step()
    avg_epoch_loss = epoch_loss / len(train_loader_sr)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss}")

    # Optionally, add validation here

    # Save checkpoint
    torch.save(model.state_dict(), f"./checkpoints/deeplabv3_ade20k_sr_epoch_{epoch+1}.pth")
