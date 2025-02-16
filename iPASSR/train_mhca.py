from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.data import DataLoader, random_split
from model_with_mhca import Net_with_MHCA
from utils import *
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from utils import *
from model import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str,
                        default='/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/train/m_files')
    parser.add_argument('--model_name', type=str, default='iPASSR_Tuned')  # Updated model name
    parser.add_argument('--load_pretrain', type=bool, default=True)
    #parser.add_argument('--model_path', type=str, default='log/iPASSR_with_MHCA_4xSR_epoch202.pth.tar')  # Path to pretrained iPASSR model
    parser.add_argument('--model_path', type=str, default='log/iPASSR_4xSR.pth.tar')  # Path to pretrained iPASSR model
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

            SR_left, SR_right = net(LR_left, LR_right, is_training=0)

            psnr_left = psnr(SR_left, HR_left)
            psnr_right = psnr(SR_right, HR_right)
            total_psnr += (psnr_left.item() + psnr_right.item()) / 2

    avg_psnr = total_psnr / len(val_loader)
    print(f'Validation PSNR: {avg_psnr:.4f} dB')
    net.train()

def train(train_loader, val_loader, cfg):
    # Initialize the new model with MHCA
    net = Net_with_MHCA(cfg.scale_factor).to(cfg.device)
    cudnn.benchmark = True
    scale = cfg.scale_factor

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            # Load the pretrained iPASSR model
            pretrained_dict = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})['state_dict']

            # Get the state dict of the new model
            model_dict = net.state_dict()

            # Filter out unnecessary keys and load pretrained parameters
            pretrained_dict = {k: v for k, v in pretrained_dict.items()} #if k in model_dict and "mhca" not in k}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

            print("Loaded pretrained parameters from '{}'".format(cfg.model_path))
        else:
            print("=> no model found at '{}'".format(cfg.model_path))

    # for name, param in net.named_parameters():
    #     if "mhca" not in name:  # Freeze all layers except MHCA layers
    #         param.requires_grad = False
    #print("Pretrained keys:", pretrained_dict.keys())
    #print("Model keys:", model_dict.keys())

    mhca_params = [p for n, p in net.named_parameters() if 'mhca' in n]
    base_params = [p for n, p in net.named_parameters() if 'mhca' not in n]
    # optimizer = torch.optim.Adam([
    #     {'params': base_params, 'lr': cfg.lr * 0.1},
    #     {'params': mhca_params, 'lr': cfg.lr}
    # ])
    criterion_L1 = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    validate(val_loader, net, cfg)

    loss_epoch = []
    loss_list = []
    print(len(train_loader))
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):

        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device), \
                Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            SR_left, SR_right, (M_right_to_left, M_left_to_right), (V_left, V_right) \
                = net(LR_left, LR_right, is_training=1)

            ''' SR Loss '''
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)

            ''' Photometric Loss '''
            Res_left = torch.abs(
                HR_left - F.interpolate(LR_left, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_left = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_right = torch.abs(
                HR_right - F.interpolate(LR_right, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_right = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                  Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                   Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                   ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_photo = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))

            ''' Smoothness Loss '''
            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h

            ''' Cycle Loss '''
            Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                       Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                       ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                        Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cycle = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1),
                                      Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1),
                                      Res_right_cycle * V_right.repeat(1, 3, 1, 1))

            ''' Consistency Loss '''
            SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale, mode='bicubic',
                                        align_corners=False)
            SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale, mode='bicubic',
                                         align_corners=False)
            SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w),
                                     SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                     ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w),
                                      SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1),
                                     SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                        criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1),
                                     SR_right_resT * V_right.repeat(1, 3, 1, 1))

            ''' Total Loss '''
            loss = loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_epoch.append(loss.data.cpu())

            if idx_iter % 300 == 0:
                print("Iterations ", idx_iter)
                validate(val_loader, net, cfg)

        scheduler.step()

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))

            print('Epoch--%4d, loss--%f, loss_SR--%f, loss_photo--%f, loss_smooth--%f, loss_cycle--%f, loss_cons--%f' %
                  (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(loss_SR.data.cpu()).mean()),
                   float(np.array(loss_photo.data.cpu()).mean()), float(np.array(loss_smooth.data.cpu()).mean()),
                   float(np.array(loss_cycle.data.cpu()).mean()), float(np.array(loss_cons.data.cpu()).mean())))
            torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                       'log/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(
                           idx_epoch + 1) + '.pth.tar')
            loss_epoch = []


def main(cfg):
    train_set = TrainSetLoader(cfg)
    val_size = 100
    train_size = len(train_set) - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    train_loader = DataLoader(dataset=train_subset, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, num_workers=6, batch_size=1, shuffle=False)
    train(train_loader, val_loader, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)