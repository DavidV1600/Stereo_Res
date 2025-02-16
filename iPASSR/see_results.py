from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
import os
from utils import *
from model import *
from model_with_mhca import *
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)  # Single image for display
    parser.add_argument('--model_name', type=str, default='iPASSR_small_lr')
    parser.add_argument('--model_path', type=str, default='log/iPASSR_with_MHCA_4xSR_epoch5.pth.tar')
    #parser.add_argument('--model_path', type=str, default='log/iPASSR_clasic_4xSR_epoch81.pth.tar')
    parser.add_argument('--trainset_dir', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/train/Normal_Patches')  # Change to actual validation set path
    
    return parser.parse_args()


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(1 / mse)


def display_and_save_images(val_loader, net, cfg):
    net.eval()

    save_dir = './predicted_results/' + cfg.model_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, (HR_left, HR_right, LR_left, LR_right) in enumerate(val_loader):
            if idx >= 10:  # Only process the first 10 images
                break

            HR_left, HR_right, LR_left, LR_right = (
                HR_left.to(cfg.device), HR_right.to(cfg.device),
                LR_left.to(cfg.device), LR_right.to(cfg.device))

            # Perform inference
            SR_left, SR_right = net(LR_left, LR_right, is_training=0)

            # Calculate PSNR
            psnr_left = psnr(SR_left, HR_left)
            psnr_right = psnr(SR_right, HR_right)
            print(f"Image {idx+1} - PSNR Left: {psnr_left.item():.4f}, PSNR Right: {psnr_right.item():.4f}")

            # Convert to PIL and save images
            scene_name = f"scene_{idx + 1}"
            SR_left_img = ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save(f'{save_dir}/{scene_name}_SR_L.png')

            SR_right_img = ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            SR_right_img.save(f'{save_dir}/{scene_name}_SR_R.png')

            # Also save the ground truth HR images
            HR_left_img = ToPILImage()(torch.squeeze(HR_left.data.cpu(), 0))
            HR_left_img.save(f'{save_dir}/{scene_name}_HR_L.png')

            HR_right_img = ToPILImage()(torch.squeeze(HR_right.data.cpu(), 0))
            HR_right_img.save(f'{save_dir}/{scene_name}_HR_R.png')

            # Display images for comparison
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(SR_left_img)
            ax[0].set_title(f"Predicted Left {scene_name}")
            ax[0].axis('off')
            ax[1].imshow(HR_left_img)
            ax[1].set_title(f"Ground Truth Left {scene_name}")
            ax[1].axis('off')
            plt.show()

            # Optional: You can do the same for the right image as well, if needed

    net.train()


def main(cfg):
    # Use the appropriate validation dataset loader
    val_set = TrainSetLoader(cfg)  # Assuming this is the validation set loader
    val_loader = DataLoader(dataset=val_set, num_workers=6, batch_size=cfg.batch_size, shuffle=False)

    # Load the pre-trained model
    #net = Net(cfg.scale_factor).to(cfg.device)
    net = Net_with_MHCA(cfg.scale_factor).to(cfg.device)
    model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])

    # Perform the display and save function
    display_and_save_images(val_loader, net, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
