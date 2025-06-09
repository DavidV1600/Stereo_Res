import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((1520, 1520)),  # Resize images to 520x520
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(           # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Define label transformations
def label_transform(label):
    # Resize the label to 520x520
    label = label.resize((1520, 1520), Image.NEAREST)  # Use nearest-neighbor interpolation
    
    # Convert PIL image to NumPy array
    label = np.array(label)
    
    # Convert NumPy array to tensor
    label = torch.tensor(label, dtype=torch.long)  # Convert to a tensor of type long
    return label

# Load the dataset with transformations
voc_train = VOCSegmentation(
    root='./data/VOC_train',
    year='2012',
    image_set='train',
    download=False,
    transform=image_transform,        # Apply image transformations
    target_transform=label_transform  # Apply label transformations
)

voc_val = VOCSegmentation(
    root='./data/VOC_val',
    year='2012',
    image_set='val',
    download=False,
    transform=image_transform,        # Apply image transformations
    target_transform=label_transform  # Apply label transformations
)


# Check the first training sample
image, label = voc_train[0]
print(f"Image shape: {image.shape}")  # Should be [3, 520, 520]
print(f"Label shape: {label.shape}")  # Should be [520, 520]
print(f"Image mean: {image.mean()}, std: {image.std()}")  # Should be close to [0, 0, 0] and [1, 1, 1] after normalization
# Create DataLoaders for training and validation sets
train_loader = DataLoader(voc_train, batch_size=1, shuffle=True)
val_loader = DataLoader(voc_val, batch_size=1, shuffle=False)