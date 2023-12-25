import sys
import os

import cv2
from pytorch_msssim import ms_ssim
import lpips
import torch
import glob
import numpy as np
from joblib import Parallel, delayed

invert_dir = sys.argv[1]
target_dir = sys.argv[2]

lpips_fn = lpips.LPIPS(net='vgg').eval().cuda()

def calc_scores(invert_file):
    invert_image = cv2.imread(invert_file, cv2.IMREAD_COLOR)
    target_file = os.path.join(target_dir, os.path.basename(invert_file))
    if not os.path.exists(target_file):
        target_file = target_file.replace('jpg', 'jpeg')
    if not os.path.exists(target_file):
        target_file = target_file.replace('jpeg', 'png')
    target_image = cv2.imread(target_file, cv2.IMREAD_COLOR)
    ssim = sum(cv2.quality.QualitySSIM_compute(invert_image, target_image)[0])/3

    mse = np.mean((invert_image/127.5 - target_image/127.5)**2)
    image1 = torch.from_numpy(invert_image).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()
    image2 = torch.from_numpy(target_image).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()
    ms_ssim_score = ms_ssim(image1, image2, data_range=255).cpu().numpy()
    lpips_score = torch.mean(lpips_fn(image1/127.5-1, image2/127.5-1)).cpu().item()
    return {'ssim': ssim, 'mse': mse, 'ms_ssim':ms_ssim_score, 'lpips': lpips_score}


scores = Parallel(n_jobs=4)(delayed(calc_scores)(f) for f in glob.glob(f'{invert_dir}/*.png'))

print('SSIM:', np.array([s['ssim'] for s in scores]).mean())
print('MSE:', np.array([s['mse'] for s in scores]).mean())
print('MS-SSIM:', np.array([s['ms_ssim'] for s in scores]).mean())
print('LPIPS:', np.array([s['lpips'] for s in scores]).mean())
