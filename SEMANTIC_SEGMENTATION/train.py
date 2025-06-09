import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as segmentation
from dataset import train_loader, val_loader
#from dataset_upscaled import train_loader #, val_loader
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define IoU calculation function
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

model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore void class
optimizer = optim.Adam(model.parameters(), lr=1e-5)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
ACUMULATION_STEP = 4

num_epochs = 5
for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_loss = 0.0

    for idx, (images, labels) in enumerate(train_loader):
        # Move data to the GPU if available
        images = images.to(device)
        
        # Extra safety check - ensure labels are in valid range
        invalid_indices = (labels > 20) & (labels != 255)
        if invalid_indices.any():
            labels[invalid_indices] = 0
        
        labels = labels.to(device)

        # After creating a batch in your training loop, you can add this debug code:
        if idx == 0:  # Just check the first batch
            print(f"Batch labels shape: {labels.shape}")
            print(f"Unique values in batch: {torch.unique(labels)}")
            # Verify all values are valid class indices
            invalid_mask = (labels > 20) & (labels != 255)
            if invalid_mask.any():
                print(f"Found {invalid_mask.sum().item()} invalid label values!")

        # Forward pass
        outputs = model(images)['out']
        loss = criterion(outputs, labels) / ACUMULATION_STEP  # Scale the loss

        # Backward pass and optimization
        loss.backward()

        epoch_loss += loss.item()

        if (idx + 1) % ACUMULATION_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate loss


        # Print loss every 10 batches
        if idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{idx}/{len(train_loader)}], Loss: {loss.item() * ACUMULATION_STEP}")

    # Update learning rate (if using a scheduler)
    scheduler.step()

    # Print average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_epoch_loss}")

    # Save the model checkpoint (optional)
    torch.save(model.state_dict(), f"./checkpoints/deeplabv4_epoch_{epoch+1}.pth")

    # Validation phase
    # model.eval()
    # val_loss = 0.0
    # val_iou = 0.0

    # with torch.no_grad():
    #     for idx, (images, labels) in enumerate(val_loader):
    #         # Move data to the GPU if available
    #         images = images.to(device)
    #         labels = labels.to(device)

    #         # Forward pass
    #         outputs = model(images)['out']
    #         loss = criterion(outputs, labels)

    #         # Accumulate loss
    #         val_loss += loss.item()

    #         # Calculate IoU
    #         preds = torch.argmax(outputs, dim=1)  # Get predicted class labels
    #         iou = calculate_iou(preds.cpu(), labels.cpu())
    #         val_iou += iou

    #         # Print loss and IoU every 10 batches
    #         if idx % 10 == 0:
    #             print(f"Validation Batch {idx}, Loss: {loss.item()}, IoU: {iou}")

    # # Compute average validation loss and IoU
    # avg_val_loss = val_loss / len(val_loader)
    # avg_val_iou = val_iou / len(val_loader)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}, Mean IoU: {avg_val_iou}")