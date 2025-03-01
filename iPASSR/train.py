from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn
import argparse
from utils import *
from model import *
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=81)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--trainset_dir', type=str,
                        default='/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/train/m_files')
    parser.add_argument('--model_name', type=str, default='iPASSR_small_lr')
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='log/iPASSR_4xSR.pth.tar')
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
    net = Net(cfg.scale_factor).to(cfg.device)
    cudnn.benchmark = True
    scale = cfg.scale_factor

    if cfg.load_pretrain and os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
        net.load_state_dict(model['state_dict'])
        cfg.start_epoch = model["epoch"]

    criterion_L1 = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    validate(val_loader, net, cfg)
    print(len(train_loader))
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            HR_left, HR_right, LR_left, LR_right = (
                HR_left.to(cfg.device), HR_right.to(cfg.device),
                LR_left.to(cfg.device), LR_right.to(cfg.device))

            SR_left, SR_right, _, _ = net(LR_left, LR_right, is_training=1)
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)

            optimizer.zero_grad()
            loss_SR.backward()
            optimizer.step()

            if idx_iter % 200 == 0:
                print(f'Iteration {idx_iter}, Loss: {loss_SR.item():.6f}')
                validate(val_loader, net, cfg)
        scheduler.step()
        print(f'Epoch {idx_epoch + 1} completed')

        torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                   f'log/{cfg.model_name}_{cfg.scale_factor}xSR_epoch{idx_epoch + 1}.pth.tar')


def main(cfg):
    train_set = TrainSetLoader(cfg)
    val_size = 50
    train_size = len(train_set) - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    train_loader = DataLoader(dataset=train_subset, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, num_workers=6, batch_size=1, shuffle=False)
    train(train_loader, val_loader, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
