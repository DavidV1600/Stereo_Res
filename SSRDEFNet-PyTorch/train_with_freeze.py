from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from utils import *
from model import *
from model_with_mhca import *
from torchvision.transforms import ToTensor
import os
import torch.nn.functional as F
from loss import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='./data/train/30_90')
    parser.add_argument('--model_name', type=str, default='SSRDEF_with_MHCA')
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='./checkpoints/SSRDEF_4xSR_epoch80.pth.tar')
    parser.add_argument('--mhca_model_path', type=str, default='./checkpoints/SSRDEF_with_MHCA_4xSR_epoch90.pth.tar')
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    return parser.parse_args()

def train(train_loader, cfg):
    source_net = SSRDEFNet_with_MHCA(cfg.scale_factor).cuda()  # Original model
    target_net = SSRDEFNet_with_MHCA(cfg.scale_factor).cuda()  # New model

    scale = cfg.scale_factor

    torch.backends.cudnn.benchmark = True

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            # Load pretrained weights
            model = torch.load(cfg.model_path)
            state_dict = model['state_dict']

            # Remove 'module.' prefix from keys (if present)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Load pretrained weights into source_net, ignoring missing keys (MHCA layers)
            source_net.load_state_dict(new_state_dict, strict=False)

            # Copy weights to target_net, ignoring missing keys (MHCA layers)
            target_net.load_state_dict(source_net.state_dict(), strict=False)

            cfg.start_epoch = model["epoch"]
            print("=> Successfully transferred weights to SSRDEFFNet_with_MHCA")
        else:
            print("=> No model found at '{}'".format(cfg.model_path))

    net = target_net

    # Freeze all layers except MHCA layers
    for name, param in net.named_parameters():
        if "mhca" not in name:  # Freeze all layers except MHCA layers
            param.requires_grad = False

    # Print the number of trainable parameters
    print(f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

    criterion_L1 = torch.nn.L1Loss().cuda()

    # Optimizer only updates MHCA layers
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad],  # Only MHCA layers
        lr=cfg.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    loss_epoch = []
    loss_list = []
    psnr_epoch = []
    psnr_epoch_r = []
    psnr_epoch_m = []
    psnr_epoch_r_m = []

    print(len(train_loader))
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            _, _, h2, w2 = HR_left.shape
            HR_left, HR_right, LR_left, LR_right = Variable(HR_left).cuda(), Variable(HR_right).cuda(), \
                Variable(LR_left).cuda(), Variable(LR_right).cuda()

            # Forward pass
            SR_left, SR_right, SR_left2, SR_right2, SR_left3, SR_right3, SR_left4, SR_right4, \
                (M_right_to_left, M_left_to_right), (disp1, disp2), (V_left, V_right), (V_left2, V_right2), (
            disp1_high, disp2_high), \
                (M_right_to_left3, M_left_to_right3), (disp1_3, disp2_3), (V_left3, V_right3), (V_left4, V_right4), (
            disp1_high_2, disp2_high_2) \
                = net(LR_left, LR_right, is_training=1)

            # Compute losses
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right) + criterion_L1(SR_left2, HR_left) + criterion_L1(SR_right2, HR_right) + \
                      criterion_L1(SR_left3, HR_left) + criterion_L1(SR_right3, HR_right) + criterion_L1(SR_left4, HR_left) + criterion_L1(SR_right4, HR_right)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_SR.backward()
            optimizer.step()

            # Logging and evaluation
            psnr_epoch.append(cal_psnr(HR_left[:, :, :, 30:].data.cpu(), SR_left4[:, :, :, 30:].data.cpu()))
            psnr_epoch_r.append(cal_psnr(HR_right[:, :, :, :HR_right.shape[3] - 30].data.cpu(), SR_right4[:, :, :, :HR_right.shape[3] - 30].data.cpu()))
            loss_epoch.append(loss_SR.data.cpu())

            if idx_iter % 300 == 0:
                print(f"Epoch {idx_epoch + 1}, Iter {idx_iter}, Loss: {loss_SR.item()}")

        scheduler.step()

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(f'Epoch {idx_epoch + 1}, Loss: {float(np.array(loss_epoch).mean())}')
            print(f'PSNR left: {float(np.array(psnr_epoch).mean())}, PSNR right: {float(np.array(psnr_epoch_r).mean())}')
            loss_epoch = []
            psnr_epoch = []
            psnr_epoch_r = []

        # Save checkpoint
        torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                   f'checkpoints/{cfg.model_name}_{cfg.scale_factor}xSR_epoch{idx_epoch + 1}.pth.tar')

def main(cfg):
    train_set = TrainSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)