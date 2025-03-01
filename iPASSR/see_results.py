from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
import os
from utils import *
from model import *
from model_with_mhca import *
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)  # Single image for display
    #parser.add_argument('--model_name', type=str, default='iPASSR_original')
    parser.add_argument('--model_name', type=str, default='iPASSR_custom_6')
    parser.add_argument('--model_path', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/iPASSR/log/iPASSR_MMHCA4_4xSR_iter61501.pth.tar')
    #parser.add_argument('--model_path', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/iPASSR/log/iPASSR_4xSR.pth.tar')
    parser.add_argument('--left_input_image', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/test/Middlebury/lr_x2/motorcycle/lr0.png')  # Path to the left HR input image
    parser.add_argument('--right_input_image', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/test/Middlebury/lr_x2/motorcycle/lr1.png')  # Path to the right HR input image
    return parser.parse_args()


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(1 / mse)


def extract_patch(image, patch_size=(100, 300)):
    """
    Extracts a patch of size `patch_size` from the center of the image.
    """
    h, w = image.size
    start_h = (h - patch_size[0]) // 2 - 20
    start_w = (w - patch_size[1]) // 2 + 80
    patch = image.crop((start_w, start_h, start_w + patch_size[1], start_h + patch_size[0]))
    return patch


def display_and_save_images(net, cfg):
    net.eval()

    save_dir = './predicted_results/' + cfg.model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the left and right HR input images
    hr_left_image = Image.open(cfg.left_input_image).convert('RGB')
    hr_right_image = Image.open(cfg.right_input_image).convert('RGB')
    print(f"Left HR image size: {hr_left_image.size}")
    print(f"Right HR image size: {hr_right_image.size}")

    # Extract a 100x300 patch from the HR images
    patch_size = (40, 120)
    hr_left_patch = extract_patch(hr_left_image, patch_size)
    hr_right_patch = extract_patch(hr_right_image, patch_size)
    print(f"Extracted left patch size: {hr_left_patch.size}")
    print(f"Extracted right patch size: {hr_right_patch.size}")

    # Downscale the patches to create the LR inputs
    lr_left_patch = hr_left_patch.resize((patch_size[1] // cfg.scale_factor, patch_size[0] // cfg.scale_factor), Image.BICUBIC)
    lr_right_patch = hr_right_patch.resize((patch_size[1] // cfg.scale_factor, patch_size[0] // cfg.scale_factor), Image.BICUBIC)
    print(f"LR left patch size: {lr_left_patch.size}")
    print(f"LR right patch size: {lr_right_patch.size}")

    # Convert patches to tensors
    hr_left_patch_tensor = ToTensor()(hr_left_patch).unsqueeze(0).to(cfg.device)
    hr_right_patch_tensor = ToTensor()(hr_right_patch).unsqueeze(0).to(cfg.device)
    lr_left_patch_tensor = ToTensor()(lr_left_patch).unsqueeze(0).to(cfg.device)
    lr_right_patch_tensor = ToTensor()(lr_right_patch).unsqueeze(0).to(cfg.device)

    with torch.no_grad():
        # Perform inference
        SR_left, SR_right = net(lr_left_patch_tensor, lr_right_patch_tensor, is_training=0)

        # Normalize outputs to [0, 1] if necessary
        SR_left = (SR_left - SR_left.min()) / (SR_left.max() - SR_left.min())
        SR_right = (SR_right - SR_right.min()) / (SR_right.max() - SR_right.min())

        # Convert to PIL images
        SR_left_img = ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0).clamp(0, 1))
        SR_right_img = ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0).clamp(0, 1))

        # Save the results
        scene_name = os.path.splitext(os.path.basename(cfg.left_input_image))[0]
        SR_left_img.save(f'{save_dir}/{scene_name}_SR_L.png')
        SR_right_img.save(f'{save_dir}/{scene_name}_SR_R.png')
        hr_left_patch.save(f'{save_dir}/{scene_name}_HR_left_patch.png')
        hr_right_patch.save(f'{save_dir}/{scene_name}_HR_right_patch.png')
        lr_left_patch.save(f'{save_dir}/{scene_name}_LR_left_patch.png')
        lr_right_patch.save(f'{save_dir}/{scene_name}_LR_right_patch.png')

        # # Display images for comparison
        # fig, ax = plt.subplots(2, 3, figsize=(18, 12))
        # ax[0, 0].imshow(lr_left_patch)
        # ax[0, 0].set_title(f"LR Left Patch (Input)")
        # ax[0, 0].axis('off')
        # ax[0, 1].imshow(hr_left_patch)
        # ax[0, 1].set_title(f"HR Left Patch (Ground Truth)")
        # ax[0, 1].axis('off')
        # ax[0, 2].imshow(SR_left_img)
        # ax[0, 2].set_title(f"Predicted Left")
        # ax[0, 2].axis('off')
        # ax[1, 0].imshow(lr_right_patch)
        # ax[1, 0].set_title(f"LR Right Patch (Input)")
        # ax[1, 0].axis('off')
        # ax[1, 1].imshow(hr_right_patch)
        # ax[1, 1].set_title(f"HR Right Patch (Ground Truth)")
        # ax[1, 1].axis('off')
        # ax[1, 2].imshow(SR_right_img)
        # ax[1, 2].set_title(f"Predicted Right")
        # ax[1, 2].axis('off')
        # plt.tight_layout()
        # plt.show()

    net.train()


def main(cfg):
    # Load the pre-trained model
    net = Net_with_MHCA(cfg.scale_factor).to(cfg.device)
    #net = Net(cfg.scale_factor).to(cfg.device)
    model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])

    # Perform the display and save function
    display_and_save_images(net, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)