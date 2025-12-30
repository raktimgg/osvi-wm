import copy
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
from models.decoder import TransposedConvDecoder

from models.attentive_pooler import AttentivePooler
from models.model import StateSpaceModel
from dataset.agent_teacher_dataset import AgentTeacherDataset
from losses.loss import compute_infonce_loss, off_diag_cov_loss, sdtw_feature_loss, sdtw_trajectory_loss
from models.decoder import PerceptualLoss


def main(args):
    device = 'cuda'

    # Extract variables directly with dictionary indexing
    data_config = args["data"]
    data_config['test_tasks'] = ["button-press-v2"]
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
    num_epochs = 10
    use_accelerate = training["use_accelerate"]
    log_wandb = training["log_wandb"]
    ema = training["ema"]
    # batch_size = training["batch_size"]
    batch_size = 8
    num_workers=training["num_workers"]
    mixup = training["mixup"]

    if 'metaworld' in data_config['agent_dir']:
        metaworld = True
    else:
        metaworld = False

    # print(metaworld)
    # data_config['novar'] = True

    # use_accelerate = False

    # print('Using T_pair', data_config['T_pair'])
    # print('Ablation: Not using Stop Grad')
    

    if use_accelerate:
        # accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        accelerator = Accelerator()

    if log_wandb and use_accelerate and accelerator.is_local_main_process:
        wandb.login()
        name = "Trial_NoSG_"+str(np.random.randint(1000))
        run = wandb.init(
            # Set the project where this run will be logged
            # project="ZSIM-Downstream-Metaworld-AWDA",
            project="ZSIM-Downstream-Metaworld-PP",
            # project="ZSIM-Downstream-Franka",
            # project="ZSIM-Downstream-HumanFranka",
            # project="ZSIM-Downstream-Mosaic",
            # Track hyperparameters and run metadata
            name=name,
        )

    train_data_config = copy.deepcopy(data_config)
    train_data_config['T_context'] = 20
    train_data_config['T_pair'] = 20
    train_dataset = AgentTeacherDataset(**train_data_config, mode='test')
    # aux_configs = [{**data_config,**aux} for aux in [data_aux_config]][0]
    # aux_train_dataset = AgentTeacherDataset(**aux_configs, mode='train')
    # full_train_dataset = torch.utils.data.ConcatDataset([train_dataset, aux_train_dataset])
    full_train_dataset = train_dataset
    val_dataset = AgentTeacherDataset(**data_config, mode='test')
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    subset_size = int(len(val_dataset) * 0.05)
    subset_indices = np.random.choice(len(val_dataset), subset_size, replace=False)

    # Create a temporary DataLoader with the subset
    small_val_loader = torch.utils.data.DataLoader(Subset(val_dataset, subset_indices), 
                                                batch_size=val_loader.batch_size, 
                                                shuffle=False, 
                                                num_workers=val_loader.num_workers)
    

    ss_model = StateSpaceModel(latent_dim=latent_dim, out_dim=out_dim, waypoints=waypoints, 
                            sub_waypoints=sub_waypoints, ent_head=ent_head, num_ent_head=num_ent_head, metaworld=metaworld).to(device)


    # ss_model._to_embed = torch.nn.Sequential(torch.nn.Linear(1024*5, latent_dim)).to(device)

    # checkpoint = torch.load('checkpoints/pre_train_awda/model_seq_9.pt', map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
    checkpoint = torch.load('checkpoints/downstream_train_awda/model_seq_alpha1_5.pt', map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
    pretrained_dict = checkpoint
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = v
    ss_model.load_state_dict(new_state_dict)
    print("Loaded pretrained weights for the ss_model.") 

    ss_model.eval()


    decoder = TransposedConvDecoder(observation_shape=(3, 224, 224), 
                                    emb_dim=512*2, activation=nn.ReLU, 
                                    depth=32, kernel_size=5, stride=3).to(device)



    optimizer = torch.optim.AdamW([
        {'params': decoder.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-5)
    # ], weight_decay=1e-12)

    # Define the learning rate scheduler with the same optimizer
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        pct_start=0.0,
        final_div_factor=1e1,
        max_lr=[1e-3],
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        cycle_momentum=False
    )

    
    criterion_mse = nn.MSELoss()
    p_loss = PerceptualLoss(device=device)
    
    scaler = torch.amp.GradScaler('cuda')

    if use_accelerate:
        decoder, ss_model, optimizer, train_loader, val_loader, small_val_loader, lr_scheduler = accelerator.prepare(
        decoder, ss_model, optimizer, train_loader, val_loader, small_val_loader, lr_scheduler)

    if use_accelerate and accelerator.is_local_main_process and log_wandb:
        wandb.watch(decoder, log_freq=1)

    def train_fn(ss_model, decoder, train_loader, val_loader, 
                 lr_scheduler, optimizer, train, small_val=False):
        ss_model.eval()
        if train:
            decoder.train()
            loader = train_loader
        else:
            decoder.eval()
            loader = val_loader
        tot_loss = 0

        if use_accelerate:
            if accelerator.is_local_main_process and not small_val:
                iterator = tqdm(loader, dynamic_ncols=True, leave=True)
            else:
                iterator = loader
        else:
            iterator = tqdm(loader, dynamic_ncols=True, leave=True)
        for i, (expert_context, agent_traj, labels) in enumerate(iterator):
            images = agent_traj['images'].to(device)
            context = expert_context['video'].to(device)
            traj_points = agent_traj['traj_points'].to(device)
            projection_matrix = agent_traj['projection_matrix'].to(device)
            head_label = agent_traj['head_label'].to(device)
            basegam = 1e-3
        
            # # # plot the first five frames of the first clip in each batch for agent and expert for debugging
            # for j in range(context.shape[1]):
            #     img_expert = context[0, j, :, :, :].cpu().detach().numpy().transpose(1, 2, 0)
            #     # denormalize images with mean and std
            #     MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
            #     STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
            #     img_expert = img_expert*STD + MEAN
            #     cv2.imwrite(f"images/expert_{i}_{j}.png", img_expert*255)
            
            # continue
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.inference_mode():
                    out = ss_model(None, images, context, pre_train=False,ents=head_label)
                if train:
                    B, T1, C, H, W = context.shape
                    B, T2, C, H, W = images.shape
                    _,T3,F = out['enc_features'].shape
                    all_features = out['enc_features'].clone().view(B*T3,F)
                    # print(torch.mean(all_features), torch.std(all_features), torch.max(all_features), torch.min(all_features))
                    all_features = all_features+torch.randn(size=all_features.shape).to(device)*1.0
                    all_images = torch.cat([context, images], dim=1).view(B*(T1+T2), C, H, W)
                    # B, T1, C, H, W = context.shape
                    # B, T2, C, H, W = images.shape
                    # _,T3,F,H2,W2 = out['raw_forward_states'].shape
                    # all_features = out['raw_forward_states'].clone().view(B*T3,F,H2,W2)
                    # all_images = images[:,1:].reshape(B*(T2-1), C, H, W)
                    im_out = decoder(all_features)
                else:
                    B, T1, C, H, W = context.shape
                    B, T2, C, H, W = images.shape
                    _,T3,F = out['forward_states'].shape
                    all_features = out['forward_states'].clone().view(B*T3,F)
                    all_images = images[:,1:].reshape(B*(T2-1), C, H, W)
                    with torch.inference_mode():
                        im_out = decoder(all_features)

                loss = criterion_mse(im_out, all_images)
                # loss2 = loss + 0.1*p_loss(im_out, all_images)
                tot_loss += loss.detach().cpu().numpy()

            if train:
                if use_accelerate:
                    accelerator.backward(scaler.scale(loss))
                else:
                    scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
            if not small_val:
                if use_accelerate:
                    if accelerator.is_local_main_process:
                        iterator.set_postfix(loss=f'{loss.detach().cpu().numpy():.3f}')
                else:
                    iterator.set_postfix(loss=f'{loss.detach().cpu().numpy():.3f}')
            if train and (i)%100==0: 
            # if train and (i+1)%22==0: # for franka
                small_val_loss = train_fn(ss_model, decoder, train_loader, small_val_loader, lr_scheduler, optimizer, train=False, small_val=True)
                decoder.train()
                if use_accelerate:
                    small_val_loss = accelerator.gather(torch.tensor(small_val_loss, device=device))
                    small_val_loss = small_val_loss.mean(dim=0).cpu().numpy() / len(small_val_loader)
                    # train_loss = accelerator.gather(torch.tensor(tot_loss, device=device))
                    # train_loss = train_loss.mean(dim=0).cpu().numpy() / (i+1)
                    train_loss = loss.detach().cpu().numpy()
                    current_lr = optimizer.param_groups[0]['lr']
                    if accelerator.is_local_main_process:
                        print(f"Training Loss: [{train_loss:.6f}]")
                        print(f"Validation Loss: [{small_val_loss:.6f}]")
                        if log_wandb:
                            wandb.log({"Train Loss": train_loss, 
                                        "Small Validation Loss": small_val_loss, 
                                    "LR": current_lr})
        return tot_loss

    for epoch in range(num_epochs):
        if use_accelerate and accelerator.is_local_main_process:
            print(f"Epoch {epoch}")
        elif not use_accelerate:
            print(f"Epoch {epoch}")
        train_loss = train_fn(ss_model, decoder, train_loader, val_loader, lr_scheduler, optimizer, train=True)
        val_loss = train_fn(ss_model, decoder, train_loader, val_loader, lr_scheduler, optimizer, train=False)

        ##########################################################
        ##################### LOGGING ############################
        ##########################################################
        if use_accelerate:
            train_loss = accelerator.gather(torch.tensor(train_loss, device=device))
            train_loss = train_loss.mean(dim=0).cpu().numpy() / len(train_loader)

            val_loss = accelerator.gather(torch.tensor(val_loss, device=device))
            val_loss = val_loss.mean(dim=0).cpu().numpy() / len(val_loader)

            if accelerator.is_local_main_process:
                
                train_loss_str = f"{train_loss:.6f}"

                val_loss_str = f"{val_loss:.6f}"

                current_lr = optimizer.param_groups[0]['lr']
                print(f"Training Loss: [{train_loss_str}]")
                print(f"Validation Loss: [{val_loss_str}]")
                print(f"Current LR: [{current_lr}]")

                state = {
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    # 'momentum_scheduler_state': list(momentum_scheduler),  # Convert the generator to a list
                }

                # Save the dictionary
                torch.save(state, "checkpoints/downstream_train_awda/decoder2.pt")
                # torch.save(state, "checkpoints/downstream_train_franka/model_seq4.pt")
                # torch.save(state, "checkpoints/downstream_train_pp/model_nosg.pt")
                print(f"Model saved!")
        else:
            train_loss /= len(train_loader)

            val_loss /= len(val_loader)

            train_loss_str = f"{train_loss:.6f}"
            val_loss_str = f"{val_loss:.6f}"
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Training Loss: [{train_loss_str}]")
            print(f"Current LR: [{current_lr}]")
            print(f"Validation Loss: [{val_loss_str}]")

            torch.save(decoder.state_dict(), "checkpoints/downstream_train_awda/decoder.pt")
            # torch.save(model.state_dict(), "checkpoints/downstream_train_pp/model_nosg.pt")
            print(f"Model saved!")
            


if __name__=="__main__":
    fname = 'configs/metaworld_data.yaml'
    # fname = 'configs/pick_place_data.yaml'
    # fname = 'configs/franka_data.yaml'
    # fname = 'configs/human_franka_data.yaml'
    # fname = 'configs/mosaic_data.yaml'
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(params)
    main(params)