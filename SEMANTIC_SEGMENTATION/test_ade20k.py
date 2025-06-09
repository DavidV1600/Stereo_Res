import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from ade20k_dataset import image_transform, mask_transform
from torch.utils.data import DataLoader
from ade20k_dataset import ADE20KDataset
import numpy as np
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 150

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

def evaluate(model, dataloader):
    model.eval()
    total_iou = 0.0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device).long()
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds.cpu(), labels.cpu())
            total_iou += iou
            total_loss += loss.item()
            if idx % 10 == 0:
                print(f"Batch {idx}, Loss: {loss.item()}, IoU: {iou}")
    avg_iou = total_iou / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss}, Mean IoU: {avg_iou}")
    return avg_iou

if __name__ == "__main__":
    # Set up validation dataset and loader
    val_dataset = ADE20KDataset(
        image_dir='./data4/Ade20k/ADEChallengeData2016/images/validation',
        mask_dir='./data4/Ade20k/ADEChallengeData2016/annotations/validation',
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Test original model
    model = segmentation.deeplabv3_resnet18(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('./checkpoints/deeplabv3_ade20k_epoch_1.pth', map_location=device))
    model = model.to(device)
    print("Evaluating model trained on original ADE20K...")
    evaluate(model, val_loader)

    # Test upscaled model
    model_sr = segmentation.deeplabv3_resnet18(weights=None, num_classes=NUM_CLASSES)
    model_sr.load_state_dict(torch.load('./checkpoints/deeplabv3_ade20k_sr_epoch_1.pth', map_location=device))
    model_sr = model_sr.to(device)
    print("Evaluating model trained on upscaled ADE20K...")
    evaluate(model_sr, val_loader)
