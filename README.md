# hypershpere-gan-inversion
This repository contains code for the paper Revisiting Latent Space of GAN Inversion for Robust Real Image Editing (WACV 2024) 
[Paper](https://openaccess.thecvf.com/content/WACV2024/html/Katsumata_Revisiting_Latent_Space_of_GAN_Inversion_for_Robust_Real_Image_WACV_2024_paper.html)

This repo is implemented upon the [BDInvert repo](https://github.com/kkang831/BDInvert_Release), [HGFI3D repo](https://github.com/jiaxinxie97/HFGI3D), [EG3D repo](https://github.com/NVlabs/eg3d), [PTI repo](https://github.com/danielroich/PTI), and[Deep3DFaceRecon repo](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21).

The main dependencies for BDInvert are:
- Python 3.9.7 or later
- PyTorch 1.10 or later
- TensorFlow 2.4.1 with GPU support 

The main dependencies for HFGI3D are:
- Python 3.9.12 or later
- PyTorch 1.12.1 or later
- TensorFlow 2.13.0 with GPU support 


## Setup enviroment for BDInvert and HGFI3D

```bash
conda env create BDInvert/environment.yaml
conda activate BDInvert
```

```bash
conda env create HFGI3D/environment.yaml
conda activate BDInvert
```

## Inversion with BDInvert

### Download pretrained base code encoder.
Download and unzip  under `BDInvert/pretrained_models/`.

| Encoder Pretrained Models                   | Basc Code Spatial Size |
| :--                                         | :--    |
| [StyleGAN2 pretrained on FFHQ 1024, 16x16](https://drive.google.com/file/d/1Gwi7I72vL7rdwET1Q0QnR71ZuZ0M3Jx1/view?usp=sharing)    | 16x16


### Inversion

Make image list.
```shell
python make_list.py --image_folder ./test_img
```

Embed images into StyleGAN's latent codes.
```shell
python invert_zp.py --job_name stylegan2_ffhq1024_zp --image_list test_img/test.list --encoder_pt_path ../pretrained_weight/encoder_stylegan2_ffhq1024_basesize16.pth | tee logs/invert_zp.txt
```

### Editing
```bash
python3 edit_ganspace_zp.py work_dirs/inversion/stylegan2_ffhq1024_zp --edit_direction ./editings/ganspace_directions/stylegan2-ffhq_style_ipca_c80_n1000000.npz --start_distance -3.0 --end_distance 3.0
```

## Inversion with HGFI3D


```bash
PYTHONPATH='./' python3 scripts/run_pti.py ../example_configs/config.py
```
