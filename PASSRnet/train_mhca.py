from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from Mhca import MHCA
from model_mhca import *
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torch.utils.data import ConcatDataset

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=5, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/train/m_files')
    parser.add_argument('--model_name', type=str, default='PASSR_MHCA4')  # Updated model name
    parser.add_argument('--load_pretrain', type=bool, default=True)
    #parser.add_argument('--model_path', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/PASSRnet/log/PASSR_MHCA4_4xSR_iter21501.pth.tar')  # Path to pretrained iPASSR model
    parser.add_argument('--model_path', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/PASSRnet/log/x4/PASSRnet_x4.pth')  # Path to pretrained iPASSR model
    return parser.parse_args()

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(1 / mse)


def validate(val_loader, net, cfg):
    net.eval()
    total_psnr = 0
    with torch.no_grad():
        for HR_left, HR_right, LR_left, LR_right in val_loader:
            HR_left, HR_right, LR_left, LR_right = (
                HR_left.to(cfg.device), HR_right.to(cfg.device),
                LR_left.to(cfg.device), LR_right.to(cfg.device))

            SR_left = net(LR_left, LR_right, is_training=0)

            psnr_left = psnr(SR_left, HR_left)
            total_psnr += psnr_left

    avg_psnr = total_psnr / len(val_loader)
    print(f'Validation PSNR: {avg_psnr:.4f} dB')
    net.train()

def train(train_loader, val_loader, cfg):
        # Initialize the new model with MHCA
    old_net = PASSRnet(cfg.scale_factor).to(cfg.device)
    #old_net = PASSRnet_with_MHCA(cfg.scale_factor).to(cfg.device)

    net = PASSRnet_with_MHCA(cfg.scale_factor).to(cfg.device)
    cudnn.benchmark = True
    scale = cfg.scale_factor

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            pretrained_dict = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})  # Load the state dictionary directly
            #old_net.load_state_dict(pretrained_dict["state_dict"])
            old_net.load_state_dict(pretrained_dict)  # Load the weights into the old model

            # Get the state dict of the new model
            model_dict = net.state_dict()

            # Filter out unnecessary keys and load pretrained parameters
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # Update the new model's state dictionary with the pretrained weights
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

            print("Loaded pretrained parameters from '{}'".format(cfg.model_path))
        else:
            print("=> no model found at '{}'".format(cfg.model_path))

    # for name, param in net.named_parameters():
    #     if "mhca" not in name:  # Freeze all layers except MHCA layers
    #         param.requires_grad = False
    # print("Pretrained keys:", pretrained_dict.keys())
    # print("Model keys:", model_dict.keys())

    # mhca_params = [p for n, p in net.named_parameters() if 'mhca' in n]
    # base_params = [p for n, p in net.named_parameters() if 'mhca' not in n]
    # optimizer = torch.optim.Adam([
    #     {'params': base_params, 'lr': cfg.lr * 0.01},
    #     {'params': mhca_params, 'lr': cfg.lr}
    # ])
    criterion_L1 = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    print("Initial model:")
    validate(val_loader, old_net, cfg)
    print(len(train_loader))

    cudnn.benchmark = True
    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            HR_left, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            SR_left, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = net(LR_left, LR_right, is_training=1)

            ### loss_SR
            loss_SR = criterion_mse(SR_left, HR_left)

            ### loss_smoothness
            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h

            ### loss_cycle
            Identity = Variable(torch.eye(w, w).repeat(b, h, 1, 1), requires_grad=False).to(cfg.device)
            loss_cycle = criterion_L1(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3), Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
                         criterion_L1(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3), Identity * V_right_to_left.permute(0, 2, 1, 3))

            ### loss_photometric
            LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b*h,w,w), LR_right.permute(0,2,3,1).contiguous().view(b*h, w, c))
            LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
            LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

            loss_photo = criterion_L1(LR_left * V_left_to_right, LR_right_warped * V_left_to_right) + \
                          criterion_L1(LR_right * V_right_to_left, LR_left_warped * V_right_to_left)

            ### losses
            loss = loss_SR + 0.005 * (loss_photo + loss_smooth + loss_cycle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            loss_epoch.append(loss.data.cpu())

            if idx_iter % 300 == 0 and idx_iter > 2:
                print("ITERATION: ", idx_iter)
                validate(val_loader, net, cfg)

                torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                        'log/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_iter' + str(idx_epoch + 1) + str(
                            idx_iter + 1) + '.pth.tar')
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch----%5d, loss---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

def main(cfg):
    train_set = TrainSetLoader(cfg.trainset_dir, cfg.scale_factor)

    #train_set2 = TrainSetLoader(cfg.trainset_dir2, cfg.scale_factor)
    #train_set = ConcatDataset([train_set, train_set2])

    val_size = 500
    train_size = len(train_set) - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    train_loader = DataLoader(dataset=train_subset, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, num_workers=6, batch_size=1, shuffle=False)
    train(train_loader, val_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

