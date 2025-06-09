import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class ADE20KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_files = [f.replace('.jpg', '.png') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Example transforms
image_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def mask_transform(mask):
    mask = mask.resize((1024, 1024), Image.NEAREST)
    mask = np.array(mask)
    # Set ignore
    mask[mask == 0] = 255
    # Set any value > 150 to 255 (ignore)
    mask[mask > 150] = 255
    # Now subtract 1 from all non-ignore pixels
    mask = np.where(mask != 255, mask - 1, 255)
    mask = mask.astype(np.uint8)
    mask = torch.tensor(mask, dtype=torch.long)
    return mask

# For original ADE20K
train_dataset = ADE20KDataset(
    image_dir='./data4/Ade20k/ADEChallengeData2016/images/training',
    mask_dir='./data4/Ade20k/ADEChallengeData2016/annotations/training',
    image_transform=image_transform,
    mask_transform=mask_transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# For upscaled ADE20K
train_dataset_sr = ADE20KDataset(
    image_dir='./data4/ade20k_sr_4x/images/training',
    mask_dir='./data4/ade20k_sr_4x/annotations/training',
    image_transform=image_transform,
    mask_transform=mask_transform
)
train_loader_sr = DataLoader(train_dataset_sr, batch_size=2, shuffle=True)

# Just for debugging, load a few masks
dataset = ADE20KDataset(
    image_dir='./data4/Ade20k/ADEChallengeData2016/images/training',
    mask_dir='./data4/Ade20k/ADEChallengeData2016/annotations/training'
)
for i in range(5):
    _, mask = dataset[i]
    mask_np = np.array(mask)
    print(f"Sample {i} unique values:", np.unique(mask_np))
