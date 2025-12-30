import os
import random

import cv2
from einops import repeat
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
import argparse
import pprint
import yaml
from tqdm import tqdm
from PIL import Image
import wandb
from einops.layers.torch import Rearrange, Reduce
from accelerate import Accelerator, DistributedDataParallelKwargs
from models.traj_embed import TemporalPositionalEncoding

from models.attentive_pooler import AttentivePooler
from models.model import StateSpaceModel
from dataset.agent_teacher_dataset import AgentTeacherDataset
from losses.loss import compute_infonce_loss, off_diag_cov_loss, sdtw_feature_loss, sdtw_trajectory_loss



def main(args):
    device = 'cuda'

    # Extract variables directly with dictionary indexing
    data_config = args["data"]
    if 'data_aux' in args:
        data_aux_config = args["data_aux"]
    model_config = args["model"]
    training = args["training"]

    latent_dim = model_config["latent_dim"]
    out_dim = model_config["out_dim"]
    waypoints = model_config["waypoints"]
    sub_waypoints = model_config["sub_waypoints"]
    ent_head = model_config["ent_head"]
    num_ent_head = model_config["num_ent_head"]

    num_epochs = training["num_epochs"]
    use_accelerate = training["use_accelerate"]
    log_wandb = training["log_wandb"]
    ema = training["ema"]
    batch_size = training["batch_size"]
    num_workers=training["num_workers"]
    mixup = training["mixup"]
    mixup = False

    # data_config['novar'] = True

    # use_accelerate = False

    print('Using T_pair', data_config['T_pair'])

    val_dataset = AgentTeacherDataset(**data_config, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    

    model = StateSpaceModel(latent_dim=latent_dim, out_dim=out_dim, waypoints=waypoints, 
                            sub_waypoints=sub_waypoints, ent_head=ent_head, num_ent_head=num_ent_head, metaworld=False).to(device)


    # model._to_embed = torch.nn.Sequential(torch.nn.Linear(1024*5, latent_dim)).to(device)

    # # checkpoint = torch.load('checkpoints/pre_train_awda/model_seq_9.pt', map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
    # checkpoint = torch.load('checkpoints/downstream_train_franka/model_seq4.pt', map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
    checkpoint = torch.load('checkpoints/downstream_train_human_franka_push/model.pt', map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
    pretrained_dict = checkpoint
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    print("Loaded pretrained weights for the model.") 

    criterion_sdtw = sdtw_trajectory_loss
    criterion_sdtw2 = sdtw_feature_loss
    criterion_mae = nn.L1Loss()

    def train_fn(model, train_loader, val_loader, 
                 lr_scheduler, optimizer, momentum_scheduler, train, small_val=False):
        if train:
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = val_loader
        tot_loss = 0
        tot_loss1 = 0
        tot_loss2 = 0

        iterator = tqdm(loader, dynamic_ncols=True, leave=True)

        for i, (expert_context, agent_traj, labels) in enumerate(iterator):
            alpha = 1

            images = agent_traj['images'].to(device)
            context = expert_context['video'].to(device)
            traj_points = agent_traj['traj_points'].to(device)
            projection_matrix = agent_traj['projection_matrix'].to(device)
            head_label = agent_traj['head_label'].to(device)
            basegam = 1e-3

            # if mixup:
            if False:
                mix_rate = torch.tensor(np.random.uniform(0.3,1,(images.shape[0]))).to(device).float()
                shiftinds = np.concatenate((np.arange(1,images.shape[0]),[0]))
                batch_dot = lambda x,y: torch.einsum('btchw,b -> btchw',x,y)
                instance_first_images = repeat(images[:,0:1],'b 1 c h w -> b t c h w',t=images.shape[1])
                images = batch_dot(images,mix_rate)+batch_dot(instance_first_images[shiftinds],1-mix_rate)
                context_first_images = repeat(context[:,0:1],'b 1 c h w -> b t c h w',t=context.shape[1])
                context = batch_dot(context,mix_rate)+batch_dot(context_first_images[shiftinds],1-mix_rate)
            # print(images.shape, context.shape, traj_points.shape)
        

            # # # plot the first five frames of the first clip in each batch for agent and expert for debugging
            # for j in range(context.shape[1]):
            #     img_expert = context[0, j, :, :, :].cpu().detach().numpy().transpose(1, 2, 0)
            #     # denormalize images with mean and std
                # MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
                # STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
                # img_expert = img_expert*STD + MEAN
            #     cv2.imwrite(f"images/expert_{i}_{j}.png", img_expert*255)
            
            # continue
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if train:
                    out = model(None, images, context, pre_train=False,ents=head_label)
                else:
                    with torch.inference_mode():
                        out = model(None, images, context, pre_train=False,ents=head_label)
                # print(((out['pred_features'][:,-1]-out['target_features'][:,-1])**2).sum())
                # continue
                gamma = basegam
                # print(traj_points.shape)
                xyz_gt = traj_points[:, :, :3]
                b1, t, _ = xyz_gt.shape
                xyz_gt_hom = torch.cat((xyz_gt, torch.ones(xyz_gt.shape[0], xyz_gt.shape[1], 1).to(device)), dim=2).reshape(b1*t, 4, 1)
                # print(xyz_gt_hom.shape)
                T_cam_in_world = projection_matrix[:,None,...].repeat(1,t,1,1).reshape(b1*t,4,4)
                T_world_in_cam = torch.linalg.inv(T_cam_in_world)
                # print(T_world_in_cam.shape)
                xyz_cam = T_world_in_cam@xyz_gt_hom
                xyz_cam = xyz_cam.reshape(b1, t, 4)
                xyz_cam[:,:,3] = traj_points[:,:,3]
                all_waypoints = xyz_cam
                # out['waypoints'] = all_waypoints
                all_waypoints = out['waypoints'][:,-5:]

                # K = np.array([[386.76404,   0.     , 324.62094], # two-franka-env
                #             [  0.     , 386.23938, 246.1506 ],
                #             [  0.     ,   0.     ,   1.     ]])
            
                K = np.array([[388.0686,   0.     , 322.3099], # human-franka-env
                            [  0.     , 387.4823, 244.7264 ],
                            [  0.     ,   0.     ,   1.     ]])
                
                MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
                STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
                # img_expert = img_expert*STD + MEAN
                for b in range(all_waypoints.shape[0]):
                    batch_waypoint = all_waypoints[b].float().detach().cpu().numpy()
                    batch_image = images[b][0].float().detach().cpu().numpy()
                    # Transpose image to (H, W, C) for plotting
                    img = batch_image.transpose(1, 2, 0)
                    img = img*STD + MEAN
                    # print(np.max(img), np.min(img))

                    # Get XYZ (ignoring 4th value in each waypoint)
                    XYZ = batch_waypoint[:, :3]  # shape (5, 3)

                    # Project 3D to 2D using intrinsic matrix K
                    uv = []
                    for point in XYZ:
                        X, Y, Z = point
                        if Z != 0:
                            x = X
                            y = Y
                            uv_homogeneous = K @ [x, y, 1]
                            u, v = uv_homogeneous[0], uv_homogeneous[1]
                            uv.append((u, v))
                    
                    # # Plot the image and the points
                    # plt.imshow(img)
                    # for u, v in uv:
                    #     plt.scatter(u, v, c='red', s=10)
                    # plt.title(f'Image {b} with projected waypoints')
                    # plt.axis('off')
                    # plt.imsave('plot.png', img)
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
                    img = img[:,:,::-1]
                    plt.imshow(img)
                    for ii, (u, v) in enumerate(uv):
                        plt.scatter(u, v, c='red', s=10)
                        plt.text(u + 5, v - 5, str(ii), color='yellow', fontsize=10, weight='bold')  # label number
                    plt.axis('off')
                    plt.savefig('images/plot_'+str(i)+'_'+str(b)+'.png', bbox_inches='tight', pad_inches=0)
                    plt.close()


                loss1,stats = criterion_sdtw(out['waypoints'].float(),
                        traj_points.float(),
                        projection_matrix,
                        60,
                        gamma,
                        waypoints,
                        image_waypoints=True)
            
                T_context = context.shape[1]
                # loss2 = criterion_mae(out['forward_states'][:,:-1], out['enc_features'][:,T_context+1:].detach())
                loss2 = criterion_mae(out['forward_states'], out['enc_features'][:,T_context+1:].detach())
                # loss2 = criterion_mae(out['forward_states'][:,:-1], out['target_features'])
                # loss2, stats = criterion_sdtw2(out['waypoint_states'],
                #         out['target_features'],
                #         gamma)
                # loss2 = torch.zeros_like(loss1)

                loss = loss1 + alpha*loss2
                # loss = loss1 + 0*loss2
                
                tot_loss += loss.detach().cpu().numpy()
                tot_loss1 += loss1.detach().cpu().numpy()
                tot_loss2 += loss2.detach().cpu().numpy()

        return tot_loss, tot_loss1, tot_loss2

    val_loss, val_loss1, val_loss2 = train_fn(model, None, val_loader, None, None, None, train=False)

    ##########################################################
    ##################### LOGGING ############################
    ##########################################################

    val_loss /= len(val_loader)
    val_loss_str = f"{val_loss:.6f}"

    val_loss1 /= len(val_loader)
    val_loss1_str = f"{val_loss1:.6f}"

    val_loss2 /= len(val_loader)
    val_loss2_str = f"{val_loss2:.6f}"

    print(f"Validation Loss: [{val_loss_str}]")
    print(f"Validation Loss1: [{val_loss1_str}]")
    print(f"Validation Loss2: [{val_loss2_str}]")


if __name__=="__main__":
    # fname = 'configs/metaworld_data.yaml'
    # fname = 'configs/pick_place_data.yaml'
    # fname = 'configs/franka_data.yaml'
    # fname = 'configs/human_franka_data.yaml'
    fname = 'configs/human_franka_push_data.yaml'
    # fname = 'configs/mosaic_data.yaml'
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(params)
    main(params)