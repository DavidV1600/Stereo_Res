import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from dataset import val_loader
from PIL import Image
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model = model.to(device)
model.load_state_dict(torch.load("./checkpoints/deeplabv3_epoch_1.pth", weights_only=True))  # Load the best model checkpoint
model.eval()  # Set the model to evaluation mode

# Directory to save predictions
output_dir = "./predictions"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Function to save predictions as images
def save_predictions(images, preds, output_dir, start_idx):
    """
    Save the model's predictions as images.
    """
    preds = preds.cpu().numpy()  # Convert predictions to NumPy array

    for i in range(len(preds)):
        # Convert the prediction mask to a PIL image
        pred_mask = Image.fromarray(preds[i].astype('uint8'))

        # Save the prediction mask
        output_path = os.path.join(output_dir, f"pred_{start_idx + i}.png")
        pred_mask.save(output_path)

# Number of images to save
num_images_to_save = 100
saved_count = 0

# Inference loop
with torch.no_grad():  # Disable gradient computation
    for idx, (images, _) in enumerate(val_loader):
        # Move data to the GPU if available
        images = images.to(device)

        # Forward pass
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)  # Get predicted class labels

        # Save predictions
        save_predictions(images, preds, output_dir, saved_count)
        saved_count += len(images)

        # Stop after saving the desired number of images
        if saved_count >= num_images_to_save:
            break

print(f"Saved {saved_count} predictions to {output_dir}")