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

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type
from utils.misc import bool_parser

from image_tools import preprocess, postprocess, Lanczos_resizing
from models.stylegan_basecode_encoder import encoder_simple
from pca_p_space import project_w2pN

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')

    # StyleGAN
    parser.add_argument('inversion_dir', type=str, default='', help='Latent head dir directory generated from invert.py')
    parser.add_argument('--seed', type=int, default=1,
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
    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/inversion/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--job_name', type=str, default='', help='Sub directory to save the results. If not specified, the result will be saved to {save_dir}/{model_name}')
    parser.add_argument('--image_list', type=str, default='test_img/test.list', help='target image folder path')
    parser.add_argument('--pnorm_root', type=str, default='pnorm/stylegan2_ffhq1024')

    # Settings
    parser.add_argument('--basecode_spatial_size', type=int, default=16, help='spatial resolution of basecode.')
    parser.add_argument('--encoder_cfg', type=str, default='default')

    # Hyperparameter
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_iters', type=int, default=1200)
    parser.add_argument('--weight_perceptual_term', type=float, default=10.)
    parser.add_argument('--weight_basecode_term', type=float, default=10.)
    parser.add_argument('--weight_pnorm_term', type=float, default=0.01)
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

    # Get work directory and job name.
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('work_dirs', 'inversion')
    os.makedirs(work_dir, exist_ok=True)
    job_name = args.job_name
    if job_name == '':
        job_name = f'{args.model_name}'
    os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)

    # Save current file and arguments
    current_file_path = inspect.getfile(inspect.currentframe())
    current_file_name = os.path.basename(current_file_path)
    shutil.copyfile(current_file_path, os.path.join(work_dir, job_name, current_file_name))
    with open(os.path.join(work_dir, job_name, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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

    inversion_dir = args.inversion_dir

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


    # Load Pnorm
    p_mean_latent = np.load(f'{args.pnorm_root}/mean_latent.npy')
    p_eigen_values = np.load(f'{args.pnorm_root}/eigen_values.npy')
    p_eigen_vectors = np.load(f'{args.pnorm_root}/eigen_vectors.npy')

    p_mean_latent = torch.from_numpy(p_mean_latent).cuda()
    p_eigen_values = torch.from_numpy(p_eigen_values).cuda()
    p_eigen_vectors = torch.from_numpy(p_eigen_vectors).cuda()

    # Load perceptual network
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    lpips_fn.net.requires_grad_(False)


    # Get GAN type
    stylegan_type = parse_gan_type(generator) # stylegan or stylegan2

    # Define layers used for base code
    basecode_layer = int(np.log2(args.basecode_spatial_size) - 2) * 2
    if stylegan_type == 'stylegan2':
        basecode_layer = f'x{basecode_layer-1:02d}'
    elif stylegan_type == 'stylegan':
        basecode_layer = f'x{basecode_layer:02d}'
    print('basecode_layer : ', basecode_layer)

    with torch.no_grad():
        z = torch.randn(1, generator.z_space_dim).cuda()
        w = generator.mapping(z, label=None)['w']
        wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)
        basecode = generator.synthesis(wp, randomize_noise=args.randomize_noise)[basecode_layer]

    #####################################
    # main
    #####################################
    image_list = []
    with open(args.image_list, 'r') as f:
        for line in f:
            image_list.append(line.strip())
    image_num = len(image_list)

    # Define save directory
    os.makedirs(os.path.join(work_dir, job_name, 'target_images'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name, 'invert_results'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name, 'invert_basecode'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name, 'invert_detailcode'), exist_ok=True)

    # Do inversion
    for batch_idx in tqdm(range(image_num)):

        # Read images
        image_path = image_list[batch_idx]
        image_basename = os.path.splitext(os.path.basename(image_path))[0]

        image = cv2.imread(image_path)
        image_target = torch.from_numpy(preprocess(image[np.newaxis, :], channel_order='BGR')).cuda() # torch_tensor, -1~1, RGB, BCHW
        image_target = Lanczos_resizing(image_target, (generator.resolution,generator.resolution))
        image_target_resized = Lanczos_resizing(image_target, (256,256))

        target = image_target.clone()
        target_resized = image_target_resized.clone()


        # Generate starting detail codes
        detailcode_starting = generator.truncation.w_avg.clone().detach()
        detailcode_starting = detailcode_starting.view(1, 1, -1)
        detailcode_starting = detailcode_starting.repeat(1, generator.num_layers, 1)
        detailcode = detailcode_starting.clone()
        detailcode.requires_grad_(True)
        zs = torch.randn(generator.num_layers, generator.z_space_dim).cuda()
        zs = generator.mapping.norm(zs)
        zs.requires_grad_(True)
        generator.repeat_w = False
        generator.truncation.repeat_w = False


        for filename in glob(f'{inversion_dir}/invert_results/*.png'):
            print('file_name : ', filename)
            image_basename = os.path.splitext(os.path.basename(filename))[0]
            image_list.append(image_basename)
            detailcode = np.load(f'{inversion_dir}/invert_detailcode/{image_basename}.npy')


            basecode = np.load(f'{inversion_dir}/invert_basecode/{image_basename}.npy')

            
            # Save results
            with torch.no_grad():
                w = generator.mapping(torch.from_numpy(detailcode).cuda(), None)['w']
                w = w.view(1, generator.num_layers, generator.z_space_dim)

                wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)

                x_rec = generator.synthesis(wp, randomize_noise=args.randomize_noise,
                                            basecode_layer=basecode_layer, basecode=torch.from_numpy(basecode).cuda())['image']

                rec_image = postprocess(x_rec.clone())[0]
                
                os.makedirs(os.path.join(inversion_dir, f'tmp'), exist_ok=True)
                cv2.imwrite(os.path.join(inversion_dir, 'tmp', image_basename+'.png'), rec_image)



if __name__ == '__main__':
    main()
