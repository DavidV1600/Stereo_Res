import torchvision.models as models
import torch

import torch
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


total = sum([param.nelement() for param in model.parameters()])
print('   Number of params: %.2fM' % (total / 1e6))