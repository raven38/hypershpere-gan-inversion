import os
import torch
from tqdm import tqdm
from configs import hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w, save_single_image, save_single_image_z
import PIL.Image
import imageio
import numpy as np
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import torch.nn.functional as F
import torchvision.transforms as transforms
from glob import glob
from criteria import l2_loss
def rot2euler(R):
    phi = np.arctan2(R[1,2], R[2,2])
    if phi<-np.pi/2.:
      phi = phi+2*np.pi
    theta = -np.arcsin(R[0,2])
    if theta<-np.pi/2:
      theta = theta+2*np.pi
    psi = np.arctan2(R[0,1], R[0,0])
    if psi<-np.pi/2:
      psi = psi+2*np.pi
    return np.array([phi, theta, psi])
def euler2rot(euler):
    sin, cos = np.sin, np.cos
    phi, theta, psi = euler[0], euler[1], euler[2]
    R1 = np.array([[1, 0, 0],
                   [0, cos(phi), sin(phi)],
                   [0, -sin(phi), cos(phi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)],
                   [0, 1, 0],
                   [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(psi), sin(psi), 0],
                   [-sin(psi), cos(psi), 0],
                   [0, 0, 1]])
    R = R1 @ R2 @ R3
    return R

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size



class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader,paths_config, multi_views,use_wandb):
        super().__init__(data_loader, paths_config,multi_views,use_wandb)

    def train(self):

        w_path_dir = f'{self.paths_config.embedding_base_dir}/{self.paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{self.paths_config.pti_results_keyword}', exist_ok=True)
        os.makedirs(f'{self.paths_config.experiments_output_dir}',exist_ok=True)
        use_ball_holder = True
        source_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )
        mask_transform = transforms.Compose([
             transforms.ToTensor()])
        if self.multi_views:
          teacher_imgs = []

          for img_i in range(30):
              
              teacher_img = PIL.Image.open(f'{self.paths_config.multi_views_output_dir}/{self.paths_config.name}/{img_i}.png').convert('RGB')
              teacher_img = source_transform(teacher_img)
              teacher_imgs.append(teacher_img)

          teacher_imgs = torch.stack(teacher_imgs,dim=0).to(global_config.device)
  
       
        for fname, image, pose in tqdm(self.data_loader):
            pose = pose.to(global_config.device)
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{self.paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None
            if not self.multi_views:
                if hyperparameters.use_last_w_pivots:
                    w_pivot = self.load_inversions(w_path_dir, image_name)
                elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                    w_pivot = self.calc_inversions(image, pose, image_name)
            else:
                print('load')
                w_pivot = self.load_inversions(w_path_dir, image_name)
           
            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            cam_pivot = torch.Tensor([0.,0.,0.2]).to(global_config.device)
            num_keyframes = 30
            render_poses = []
           
            for frame_idx in tqdm(range(num_keyframes)):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
                                                            cam_pivot, radius=2.7, device=global_config.device)
            
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=global_config.device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                render_poses.append(c)
     
            if self.multi_views:
               max_pti_steps = hyperparameters.max_pti_steps_multiviews
            else:
               max_pti_steps = hyperparameters.max_pti_steps
            pbar = tqdm(range(max_pti_steps+1))
            for i in pbar:
              
                generated_images,_ = self.forward(w_pivot,pose,eval=False)

                loss, l2_loss_val, loss_lpips= self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)
                if self.multi_views:
                      random_pose_id = np.random.choice(np.arange(30),size=1)
                      render_image,_= self.forward(w_pivot,render_poses[random_pose_id[0]],eval=False)
                      
                      teacher_loss,teacher_l2,teacher_lpips = self.calc_loss(render_image, teacher_imgs[random_pose_id[0]:random_pose_id[0]+1,:,:], image_name,
                                                               self.G, use_ball_holder, w_pivot)
  
               
                if self.multi_views:
                      loss = loss + teacher_loss

                self.optimizer.zero_grad()
                 
                if loss_lpips <= hyperparameters.LPIPS_value_threshold and not self.multi_views:
                  with torch.no_grad():
                    generated_images, generated_depths = self.forward(w_pivot,pose,eval=True)
                    save_depth = (generated_depths.permute(0,2,3,1)*1000.)

                    imageio.imwrite(f'{self.paths_config.experiments_output_dir}/{fname[0]}_{i:04d}'+'_depth.png',np.array(save_depth[0,:,:,0].detach().cpu().numpy(),dtype=np.uint16))
    
                    img = (generated_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                
                    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.paths_config.experiments_output_dir}/{fname[0]}_{i:04d}'+'.png')
                    
                    num_keyframes = 30
                    imgs = []
                    os.makedirs(f'{self.paths_config.experiments_output_dir}'+'/{}/{}_output/'.format(image_name,i),exist_ok=True)
                    video_out = imageio.get_writer(f'{self.paths_config.experiments_output_dir}/{fname[0]}_{i:04d}'+'.mp4', mode='I', fps=20, codec='libx264')
                    for frame_idx in tqdm(range(num_keyframes)):
                        pitch_range = 0.25
                        yaw_range = 0.35
                        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
                                                                    3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
                                                                    cam_pivot, radius=2.7, device=global_config.device)
                    
                        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=global_config.device)
                        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                        
                        img,_ = self.forward(w_pivot,c,eval=True)
                        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        
                        imageio.imwrite(f'{self.paths_config.experiments_output_dir}'+'/{}/{}_output/{}.png'.format(image_name,i,frame_idx),img[0].cpu().numpy())
                        video_out.append_data(img[0].cpu().numpy())
                    video_out.close()


                    os.makedirs(f'{self.paths_config.experiments_output_dir}'+'/{}/{}_edited/'.format(image_name,i),exist_ok=True)
                    boundary_files = list(glob(f'{self.paths_config.boundaries_dir}/*.npy'))
                    print(f'{len(boundary_files)} boundaries are found')
                    for boundary_file in boundary_files:
                        boundary_name = os.path.splitext(os.path.basename(boundary_file))[0]
                        direction = torch.from_numpy(np.load(boundary_file).reshape(-1, 512)).to(w_pivot.device)
                        for sigma in [-5, -3, -1, 1, 3, 5]:
                            for frame_idx in tqdm(range(5)):                        
                                pitch_range = 0.25
                                yaw_range = 0.35
                                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (5)),
                                                                          3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (5)),
                                                                          cam_pivot, radius=2.7, device=global_config.device)
                    
                                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=global_config.device)
                                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                                if len(direction) < 20:
                                    edited_w = w_pivot + 2 * sigma * direction
                                    img,_ = self.forward(edited_w,c,eval=True)
                                else:
                                    block_zs = []
                                    z_idx = 0
                                    zs = w_pivot.view(1, -1, 512)
                                    for res in self.G.backbone.synthesis.block_resolutions:
                                        block = getattr(self.G.backbone.synthesis, f'b{res}')
                                        block_zs.append(zs.narrow(1, z_idx, block.num_conv + block.num_torgb))
                                        z_idx += block.num_conv
                                    zs = torch.cat(block_zs, 1)
                                    zs = zs.view(-1, 512)
                                    edited_zs = zs + 10 * sigma * direction
                                    num_ws = self.G.backbone.mapping.num_ws
                                    truncation_psi = 1
                                    truncation_cutoff = 14
                                    cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0.2]), device=torch.device(global_config.device))
                                    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=2.7, device=torch.device(global_config.device))
                                    cam2world_matrix = c[:, :16].view(-1, 4, 4)
                                    intrinsics = c[:, 16:25].view(-1, 3, 3)            
                                    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), pose[:,16:]], 1)
                                    conditioning_params = conditioning_params.expand([len(zs),conditioning_params.shape[1]])
                                    self.G.backbone.mapping.num_ws = None
                                    w = self.G.backbone.mapping(edited_zs, conditioning_params)
                                    self.G.backbone.mapping.num_ws = num_ws
                                    w = w.view(1, -1, 512)
                                    neural_rendering_resolution = self.G.neural_rendering_resolution
                                    ray_origins, ray_directions = self.G.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
                                    N, M, _ = ray_origins.shape
                                    block_ws = []
                                    w_idx = 0
                                    zs = w_pivot.view(1, -1, 512)
                                    for res in self.G.backbone.synthesis.block_resolutions:
                                        block = getattr(self.G.backbone.synthesis, f'b{res}')
                                        block_ws.append(w.narrow(1, w_idx, block.num_conv + block.num_torgb))
                                        w_idx += block.num_conv + block.num_torgb

                                    x = img = None
                                    for res, cur_ws in zip(self.G.backbone.synthesis.block_resolutions, block_ws):
                                        block = getattr(self.G.backbone.synthesis, f'b{res}')
                                        x, img = block(x, img, cur_ws, noise_mode='const')        

                                    planes = img
                                    # Reshape output into three 32-channel planes
                                    planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

                                    # Perform volume rendering
                                    feature_samples, depth_samples, weights_samples = self.G.renderer(planes, self.G.decoder, ray_origins, ray_directions, self.G.rendering_kwargs) # channels last

                                    # Reshape into 'raw' neural-rendered image
                                    H = W = self.G.neural_rendering_resolution
                                    feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
                                    depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

                                    # Run superresolution to get final image
                                    rgb_image = feature_image[:, :3]
                                    img = self.G.superresolution(rgb_image, feature_image, w, noise_mode=self.G.rendering_kwargs['superresolution_noise_mode'])

                                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        
                                imageio.imwrite(f'{self.paths_config.experiments_output_dir}'+'/{}/{}_edited/{}_{}_{}.png'.format(image_name,i,frame_idx, boundary_name, sigma),img[0].cpu().numpy())
                    break

                loss.backward()
                pbar.set_postfix(loss=f'{loss.item():.5f}', lpips_loss=f'{loss_lpips.item():.5f}', l2_loss=f'{l2_loss_val.item():.5f}')
                self.optimizer.step()
                
                if (i%max_pti_steps == 0 and not self.multi_views and i!=0) or (i%max_pti_steps==0 and self.multi_views and i!=0) :
                    with torch.no_grad():
 
                        generated_images, generated_depths = self.forward(w_pivot,pose,eval=True)
                  
                        img = (generated_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                      
                        save_depth = (generated_depths.permute(0,2,3,1)*1000.)

                        imageio.imwrite(f'{self.paths_config.experiments_output_dir}/{fname[0]}_{i:04d}'+'_depth.png',np.array(save_depth[0,:,:,0].detach().cpu().numpy(),dtype=np.uint16))
                        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.paths_config.experiments_output_dir}/{fname[0]}_{i:04d}'+'.png')
                            
              
            
                if (i%max_pti_steps == 0 and not self.multi_views and i!=0) or (i%max_pti_steps==0 and self.multi_views and i!=0) :
                  with torch.no_grad():   
                    num_keyframes = 30
                    imgs = []
                    os.makedirs(f'{self.paths_config.experiments_output_dir}'+'/{}/{}_output/'.format(image_name,i),exist_ok=True)
                    video_out = imageio.get_writer(f'{self.paths_config.experiments_output_dir}/{fname[0]}_{i:04d}'+'.mp4', mode='I', fps=20, codec='libx264')
                    for frame_idx in tqdm(range(num_keyframes)):
                        pitch_range = 0.25
                        yaw_range = 0.35
                        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
                                                                    3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
                                                                    cam_pivot, radius=2.7, device=global_config.device)
                    
                        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=global_config.device)
                        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                        
                        img,_= self.forward(w_pivot,c,eval=True)
                        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                        imageio.imwrite(f'{self.paths_config.experiments_output_dir}'+'/{}/{}_output/{}.png'.format(image_name,i,frame_idx),img[0].cpu().numpy())
                        video_out.append_data(img[0].cpu().numpy())
                    video_out.close()
                   
                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])

                img_dir = f'{self.paths_config.experiments_output_dir}/{self.paths_config.input_data_id}'
                os.makedirs(img_dir, exist_ok=True)
                # if hyperparameters.first_inv_type == 'z+':                
                    # save_single_image_z(img_dir, w_pivot, self.G, image_name)
                # else:
                    # save_single_image(img_dir, w_pivot, self.G, image_name)


                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            torch.save(self.G,
                       f'{self.paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
