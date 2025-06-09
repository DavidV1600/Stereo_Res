from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, transforms
import argparse
import os
from models import *
from model_mhca import *

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/test/')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    #parser.add_argument('--model_name', type=str, default='PASSRnet_x4')
    parser.add_argument('--model_name', type=str, default='PASSR_MHCA4_4xSR_iter61501')
    return parser.parse_args()


def test(cfg):
    net = PASSRnet_with_MHCA(cfg.scale_factor).to(cfg.device)
    model = torch.load('./log/' + cfg.model_name + '.pth.tar')
    net.load_state_dict(model['state_dict'])
    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor))
    for idx in range(len(file_list)):
        LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr0.png')
        LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr1.png')
        LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
        LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
        LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
        scene_name = file_list[idx]
        print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')
        with torch.no_grad():
            SR_left = net(LR_left, LR_right, is_training=1)
            
            # Check if SR_left is a tuple
            if isinstance(SR_left, tuple):
                # Assuming the first element of the tuple is the desired tensor
                SR_left = SR_left[0]
            
            SR_left = torch.clamp(SR_left, 0, 1)
        
        save_path = './results/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L.png')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['Flickr1024', 'KITTI2012', 'KITTI2015', 'Middlebury']
    #dataset_list = ['KITTI2012']
    for i in range(len(dataset_list)):
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')