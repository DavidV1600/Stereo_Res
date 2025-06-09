import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


def cal_metrics(im0, im1, boundary=0):
    im0_crop = im0[boundary:-boundary, boundary:-boundary] if boundary else im0
    im1_crop = im1[boundary:-boundary, boundary:-boundary] if boundary else im1
    #print(im0_crop.shape, im1_crop.shape)
    return psnr(im1_crop, im0_crop), ssim(im1_crop, im0_crop, channel_axis=-1)


def evaluate():
    Method = 'PASSR'  # Method for evaluation
    factor = 4  # Upsampling factor
    stereo_boundary = 64  # Cropping left 64 pixels in the left view for evaluation
    image_boundary = 0  # Cropping image boundaries for evaluation
    Datasets = ['Flickr1024', 'KITTI2012', 'KITTI2015', 'Middlebury']
    #sDatasets = ['KITTI2012']
    GT_folder = '/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/test/'
    #ResultsPath = f'./results/{Method}_small_lr_4xSR_epoch81'
    ResultsPath = f'./results/{Method}_MMHCA3_4xSR_epoch2'
    ResultsPath = f'./results/{Method}_MHCA4_4xSR_iter61501'
    #ResultsPath = f'./results/{Method}net_x4'
    print(ResultsPath)
    for DatasetName in Datasets:
        GT_DataFolder = os.path.join(GT_folder, DatasetName, 'hr')
        print(GT_DataFolder)
        GTfiles = sorted(os.listdir(GT_DataFolder))
        print(GTfiles)

        txtName = os.path.join(ResultsPath, f'{Method}_{factor}xSR_{DatasetName}.txt')
        print(txtName)
        with open(txtName, 'w') as fp:
            pass

        psnr_left_crop_vals, ssim_left_crop_vals = [], []
        psnr_stereo_vals, ssim_stereo_vals = [], []

        for scene_name in GTfiles:
            print(f'Running Scene {scene_name} in Dataset {DatasetName}......')

            gt_left = cv2.imread(os.path.join(GT_DataFolder, scene_name, 'hr0.png'))
            gt_right = cv2.imread(os.path.join(GT_DataFolder, scene_name, 'hr1.png'))

            sr_left = cv2.imread(os.path.join(ResultsPath, DatasetName, f'{scene_name}_L.png'))
            sr_right = cv2.imread(os.path.join(ResultsPath, DatasetName, f'{scene_name}_R.png'))

            gt_left_crop = gt_left[:, stereo_boundary:, :]
            sr_left_crop = sr_left[:, stereo_boundary:, :]

            psnr_left_crop, ssim_left_crop = cal_metrics(sr_left_crop, gt_left_crop, image_boundary)
            psnr_left, ssim_left = cal_metrics(sr_left, gt_left, image_boundary)

            psnr_left_crop_vals.append(psnr_left_crop)
            ssim_left_crop_vals.append(ssim_left_crop)

            with open(txtName, 'a') as fp:
                fp.write(
                    f'\n {len(psnr_left_crop_vals):03d} \t {psnr_left:.4f} \t {ssim_left:.4f} \t \n')

        psnr_left_crop_avg = np.mean(psnr_left_crop_vals)
        ssim_left_crop_avg = np.mean(ssim_left_crop_vals)
        psnr_stereo_avg = np.mean(psnr_stereo_vals)
        ssim_stereo_avg = np.mean(ssim_stereo_vals)

        with open(txtName, 'a') as fp:
            fp.write(
                f'\n AVG \t {psnr_left_crop_avg:.4f} \t {ssim_left_crop_avg:.4f} \t {psnr_stereo_avg:.4f} \t {ssim_stereo_avg:.4f} \t \n')


if __name__ == "__main__":
    evaluate()
