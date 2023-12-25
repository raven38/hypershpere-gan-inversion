# -*- coding:utf-8 -*-

import sys
sys.path.append('..')

import os, inspect, shutil, json
from types import SimpleNamespace
import argparse
import subprocess
import csv
import cv2
import numpy as np
import random
import lpips
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from glob import glob

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image
from utils.visualizer import load_image

from image_tools import preprocess, postprocess, Lanczos_resizing

import PIL

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')

    # StyleGAN
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--model_name', type=str, default='stylegan2_ffhq1024',
                        help='Name to the pre-trained model.')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')

    # IO
    parser.add_argument('image1', type=str, default='', help='Latent head dir directory generated from invert.py')
    parser.add_argument('image2', type=str, default='', help='Latent head dir directory generated from invert.py')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')

    # Settings
    parser.add_argument('--use_FW_space', type=bool_parser, default=True)
    parser.add_argument('--basecode_spatial_size', type=int, default=16, help='spatial resolution of basecode.')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Set random seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in 'f'`models/model_zoo.py`!')
    model_config = MODEL_ZOO[args.model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.
    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,
                            randomize_noise=args.randomize_noise)
    print(f'Finish building generator.')

    # Load StyleGAN
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading StyleGAN checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    generator.requires_grad_(False)
    generator.repeat_w = False
    generator.truncation.repeat_w = False
    print(f'Finish loading StyleGAN checkpoint.')

    print('interpolation', args.image1, args.image2)

    # Get GAN type
    stylegan_type = parse_gan_type(generator) # stylegan or stylegan2

    # Define layers used for base code
    basecode_layer = int(np.log2(args.basecode_spatial_size) - 2) * 2
    if stylegan_type == 'stylegan2':
        basecode_layer = f'x{basecode_layer-1:02d}'
    elif stylegan_type == 'stylegan':
        basecode_layer = f'x{basecode_layer:02d}'
    print('basecode_layer : ', basecode_layer)

    print('file_name : ', args.image1)
    image_basename1 = os.path.splitext(os.path.basename(args.image1))[0]
    inversion_dir = '/'.join(args.image1.split('/')[:-2])
    detailcode1 = np.load(f'{inversion_dir}/invert_detailcode/{image_basename1}.npy')

    if args.use_FW_space:
      basecode1 = np.load(f'{inversion_dir}/invert_basecode/{image_basename1}.npy')
      print(basecode1.shape)

    print('file_name : ', args.image2)
    image_basename2 = os.path.splitext(os.path.basename(args.image2))[0]
    inversion_dir = '/'.join(args.image2.split('/')[:-2])
    detailcode2 = np.load(f'{inversion_dir}/invert_detailcode/{image_basename2}.npy')

    if args.use_FW_space:
      basecode2 = np.load(f'{inversion_dir}/invert_basecode/{image_basename2}.npy')
      print(basecode2.shape)

    ip_details = []
    ip_bases = []

    for i in range(args.step):
        lam = i * 1.0 / (args.step-1)
        detail = torch.zeros_like(torch.tensor(detailcode1))
        for j in range(len(detailcode1)):
          norm_z1 = normalize_2nd_moment(torch.tensor(detailcode1[j]), dim=0)
          norm_z2 = normalize_2nd_moment(torch.tensor(detailcode2[j]), dim=0)
          norm_z1 = norm_z1 / norm_z1.norm(2)
          norm_z2 = norm_z2 / norm_z2.norm(2)
          omega = torch.arccos(norm_z1@norm_z2.T)
          so = torch.sin(omega)
          detail[j] = torch.sin((1.0 - lam)*omega)/so*norm_z1 + torch.sin(lam*omega)/so*norm_z2
        ip_details.append(normalize_2nd_moment(detail))
        
        base = basecode1 * (1-lam) + basecode2 * lam
        ip_bases.append(base)

    os.makedirs(os.path.join(inversion_dir, f'interpolate_results'), exist_ok=True)
    images = []
    for idx in range(args.step):
        temp_code = torch.tensor(ip_details[idx]).type(torch.FloatTensor).cuda()
        temp_code = temp_code.view(-1, generator.z_space_dim)
        temp_code = generator.mapping.norm(temp_code)
        w = generator.mapping(temp_code, None)['w']
        w = w.view(1, generator.num_layers, generator.z_space_dim)
        wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)
        
        if args.use_FW_space:
            basecode = torch.from_numpy(ip_bases[idx]).type(torch.FloatTensor).cuda()
            image = generator.synthesis(wp, randomize_noise=args.randomize_noise,
                                        basecode_layer=basecode_layer, basecode=basecode)['image']
        else:
            image = generator.synthesis(wp, randomize_noise=args.randomize_noise)['image']
        image = postprocess(image)[0]
        cv2.imwrite(os.path.join(inversion_dir, f'interpolate_results', f'{idx:03d}.png'), image)
        images.append(image)
    W = 1024
    H = 1024
    canvas = PIL.Image.new('RGB', (W * args.step, H), 'black')
    for col_idx in range(args.step):
        canvas.paste(PIL.Image.fromarray(images[col_idx][:, :, [2, 1, 0]], 'RGB'), (W*col_idx, 0))
    canvas.save(os.path.join(inversion_dir, f'interpolate_results', 'grid.png'))

if __name__ == '__main__':
    main()
