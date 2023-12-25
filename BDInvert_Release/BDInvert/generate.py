import sys
sys.path.append('.')
sys.path.append('..')

import os, inspect, shutil, json
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
import re

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type
from utils.misc import bool_parser

from image_tools import preprocess, postprocess, Lanczos_resizing
from models.stylegan_basecode_encoder import encoder_simple
from pca_p_space import project_w2pN

def num_range(s: str):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')

    # StyleGAN
    parser.add_argument('--seeds', type=num_range,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--model_name', type=str,
                        help='Name to the pre-trained model.', default='stylegan2_ffhq1024')
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
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/inversion/` by default. '
                             '(default: %(default)s)')

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    current_file_path = inspect.getfile(inspect.currentframe())
    current_file_name = os.path.basename(current_file_path)


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
    print(f'Finish loading StyleGAN checkpoint.')

    # Get GAN type
    stylegan_type = parse_gan_type(generator) # stylegan or stylegan2

    os.makedirs(args.outdir, exist_ok=True)

    # Do inversion
    for seed_idx, seed in enumerate(args.seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, generator.z_space_dim)).float().cuda()
        # Save results
        with torch.no_grad():
            w = generator.mapping(z, None)['w']
            wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)
            x_rec = generator.synthesis(wp, randomize_noise=args.randomize_noise)['image']
            rec_image = postprocess(x_rec.clone())[0]

            cv2.imwrite(os.path.join(args.outdir, f'seed{seed:010d}.png'), rec_image)


if __name__ == '__main__':
    main()
