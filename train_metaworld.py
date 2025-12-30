import copy

import cv2
from einops import repeat

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
import pprint
import yaml
from tqdm import tqdm
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs

from models.model import StateSpaceModel
from dataset.agent_teacher_dataset import AgentTeacherDataset
from losses.loss import sdtw_trajectory_loss



def main(args):
    device = 'cuda'

    # Extract variables directly with dictionary indexing
    data_config = args["data"]
    model_config = args["model"]
    training = args["training"]

    latent_dim = model_config["latent_dim"]
    waypoints = model_config["waypoints"]
    sub_waypoints = model_config["sub_waypoints"]

    num_epochs = training["num_epochs"]
    use_accelerate = training["use_accelerate"]
    log_wandb = training["log_wandb"]
    ema = training["ema"]
    batch_size = training["batch_size"]
    num_workers=training["num_workers"]
    mixup = training["mixup"]
    max_lr = training["max_lr"]
    final_div_factor = training["final_div_factor"]
    

    if use_accelerate:
        accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        # accelerator = Accelerator()

    if log_wandb and use_accelerate and accelerator.is_local_main_process:
        wandb.login()
        name = "Train_"+str(np.random.randint(1000))
        run = wandb.init(
            # Set the project where this run will be logged
            project="OSVI-WM-Metaworld-Sep",
            # Track hyperparameters and run metadata
            name=name,
        )

    train_dataset = AgentTeacherDataset(**data_config, mode='train')
    full_train_dataset = train_dataset
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    temp_config = copy.deepcopy(data_config)
    temp_config['test_tasks'] = [data_config['test_tasks'][0]]
    val_dataset = AgentTeacherDataset(**temp_config, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    subset_size = int(len(val_dataset) * 0.05)
    subset_indices = np.random.choice(len(val_dataset), subset_size, replace=False)

    small_val_loader1 = torch.utils.data.DataLoader(Subset(val_dataset, subset_indices), 
                                                batch_size=val_loader.batch_size, 
                                                shuffle=False, 
                                                num_workers=val_loader.num_workers)
    
    temp_config = copy.deepcopy(data_config)
    temp_config['test_tasks'] = [data_config['test_tasks'][1]]
    val_dataset = AgentTeacherDataset(**temp_config, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    small_val_loader2 = torch.utils.data.DataLoader(Subset(val_dataset, subset_indices), 
                                                batch_size=val_loader.batch_size, 
                                                shuffle=False, 
                                                num_workers=val_loader.num_workers)
    
    temp_config = copy.deepcopy(data_config)
    temp_config['test_tasks'] = [data_config['test_tasks'][2]]
    val_dataset = AgentTeacherDataset(**temp_config, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    small_val_loader3 = torch.utils.data.DataLoader(Subset(val_dataset, subset_indices), 
                                                batch_size=val_loader.batch_size, 
                                                shuffle=False, 
                                                num_workers=val_loader.num_workers)
    
    temp_config = copy.deepcopy(data_config)
    temp_config['test_tasks'] = [data_config['test_tasks'][3]]
    val_dataset = AgentTeacherDataset(**temp_config, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    small_val_loader4 = torch.utils.data.DataLoader(Subset(val_dataset, subset_indices), 
                                                batch_size=val_loader.batch_size, 
                                                shuffle=False, 
                                                num_workers=val_loader.num_workers)
    
    val_dataset = AgentTeacherDataset(**data_config, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    

    model = StateSpaceModel(latent_dim=latent_dim, waypoints=waypoints, sub_waypoints=sub_waypoints).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-3)

    # Define the learning rate scheduler with the same optimizer
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        pct_start=0.0,
        final_div_factor=final_div_factor,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        cycle_momentum=False
    )

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])


    
    criterion_sdtw = sdtw_trajectory_loss
    criterion_mae = nn.L1Loss()
    
    scaler = torch.amp.GradScaler('cuda')

    if use_accelerate:
        model, optimizer, train_loader, val_loader, small_val_loader1, small_val_loader2, small_val_loader3, small_val_loader4, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, small_val_loader1, small_val_loader2, small_val_loader3, small_val_loader4, lr_scheduler)

    ipe = len(train_loader)
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs)
                          for i in range(int(ipe*num_epochs)+1))

    alpha_factor = 10
    alpha_scheduler = (np.exp(-i/ipe)*0.95+0.05
                          for i in range(int(ipe*alpha_factor)+1))
    last_epoch = 0
    for _ in range(last_epoch*ipe):
        next(alpha_scheduler)

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

        if use_accelerate:
            if accelerator.is_local_main_process and not small_val:
                iterator = tqdm(loader, dynamic_ncols=True, leave=True)
            else:
                iterator = loader
        else:
            iterator = tqdm(loader, dynamic_ncols=True, leave=True)
        for i, (expert_context, agent_traj, labels) in enumerate(iterator):
            if train:
                alpha = next(alpha_scheduler)
            else:
                alpha = 0
            images = agent_traj['images'].to(device)
            context = expert_context['video'].to(device)
            traj_points = agent_traj['traj_points'].to(device)
            projection_matrix = agent_traj['projection_matrix'].to(device)
            head_label = agent_traj['head_label'].to(device)
            basegam = 1e-3

            if mixup:
                mix_rate = torch.tensor(np.random.uniform(0.3,1,(images.shape[0]))).to(device).float()
                shiftinds = np.concatenate((np.arange(1,images.shape[0]),[0]))
                batch_dot = lambda x,y: torch.einsum('btchw,b -> btchw',x,y)
                instance_first_images = repeat(images[:,0:1],'b 1 c h w -> b t c h w',t=images.shape[1])
                images = batch_dot(images,mix_rate)+batch_dot(instance_first_images[shiftinds],1-mix_rate)
                context_first_images = repeat(context[:,0:1],'b 1 c h w -> b t c h w',t=context.shape[1])
                context = batch_dot(context,mix_rate)+batch_dot(context_first_images[shiftinds],1-mix_rate)            

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if train:
                    out = model(images, context)
                else:
                    with torch.inference_mode():
                        out = model(images, context)

                gamma = basegam
                loss1,stats = criterion_sdtw(out['waypoints'],
                        traj_points,
                        projection_matrix,
                        60,
                        gamma,
                        waypoints,
                        image_waypoints=True)      
            
                T_context = context.shape[1]
                loss2 = criterion_mae(out['forward_states'], out['enc_features'][:,T_context+1:].detach())

                loss = loss1 + alpha*loss2
                
                tot_loss += loss.detach().cpu().numpy()
                tot_loss1 += loss1.detach().cpu().numpy()
                tot_loss2 += loss2.detach().cpu().numpy()

            if train:
                if use_accelerate:
                    accelerator.backward(scaler.scale(loss))
                else:
                    scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    if use_accelerate:
                        for param_q, param_k in zip(model.module._embed.encoder.parameters(), model.module._embed.target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                    else:
                        for param_q, param_k in zip(model._embed.encoder.parameters(), model._embed.target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            if not small_val:
                if use_accelerate:
                    if accelerator.is_local_main_process:
                        iterator.set_postfix(loss=f'{loss.detach().cpu().numpy():.3f}')
                else:
                    iterator.set_postfix(loss=f'{loss.detach().cpu().numpy():.3f}')
            if train and (i)%100==0:
                _, small_val_loss1, _ = train_fn(model, train_loader, small_val_loader1, lr_scheduler, optimizer, momentum_scheduler, train=False, small_val=True)
                _, small_val_loss2, _ = train_fn(model, train_loader, small_val_loader2, lr_scheduler, optimizer, momentum_scheduler, train=False, small_val=True)
                _, small_val_loss3, _ = train_fn(model, train_loader, small_val_loader3, lr_scheduler, optimizer, momentum_scheduler, train=False, small_val=True)
                _, small_val_loss4, _ = train_fn(model, train_loader, small_val_loader4, lr_scheduler, optimizer, momentum_scheduler, train=False, small_val=True)
                model.train()
                if use_accelerate:
                    small_val_loss1 = accelerator.gather(torch.tensor(small_val_loss1, device=device))
                    small_val_loss1 = small_val_loss1.mean(dim=0).cpu().numpy() / len(small_val_loader1)

                    small_val_loss2 = accelerator.gather(torch.tensor(small_val_loss2, device=device))
                    small_val_loss2 = small_val_loss2.mean(dim=0).cpu().numpy() / len(small_val_loader2)
                    
                    small_val_loss3 = accelerator.gather(torch.tensor(small_val_loss3, device=device))
                    small_val_loss3 = small_val_loss3.mean(dim=0).cpu().numpy() / len(small_val_loader3)
                    
                    small_val_loss4 = accelerator.gather(torch.tensor(small_val_loss4, device=device))
                    small_val_loss4 = small_val_loss4.mean(dim=0).cpu().numpy() / len(small_val_loader4)

                    train_loss = loss.detach().cpu().numpy()
                    train_loss1 = loss1.detach().cpu().numpy()
                    train_loss2 = loss2.detach().cpu().numpy()
                    current_lr = optimizer.param_groups[0]['lr']
                    if accelerator.is_local_main_process:
                        print(f"Training Loss: [{train_loss:.6f}]")
                        print(f"Training Loss1: [{train_loss1:.6f}]")
                        print(f"Training Loss2: [{train_loss2:.6f}]")
                        print(f"Validation Loss1: [{small_val_loss1:.6f}]")
                        print(f"Validation Loss2: [{small_val_loss2:.6f}]")
                        print(f"Validation Loss3: [{small_val_loss3:.6f}]")
                        print(f"Validation Loss4: [{small_val_loss4:.6f}]")
                        if log_wandb:
                            wandb.log({"Train Loss": train_loss, 
                                       "Train Loss1": train_loss1,
                                        "Train Loss2": train_loss2,
                                        "Small Validation Loss1": small_val_loss1,
                                        "Small Validation Loss2": small_val_loss2,
                                        "Small Validation Loss3": small_val_loss3,
                                        "Small Validation Loss4": small_val_loss4, 
                                        "LR": current_lr,
                                        "alpha": alpha})
        return tot_loss, tot_loss1, tot_loss2

    for epoch in range(last_epoch, num_epochs):
        if use_accelerate and accelerator.is_local_main_process:
            print(f"Epoch {epoch}")
        elif not use_accelerate:
            print(f"Epoch {epoch}")
        train_loss, _, _ = train_fn(model, train_loader, val_loader, lr_scheduler, optimizer, momentum_scheduler, train=True)
        val_loss, _, _ = train_fn(model, train_loader, val_loader, lr_scheduler, optimizer, momentum_scheduler, train=False)

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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }

                torch.save(state, "checkpoints/metaworld/model_"+str(epoch)+".pt")
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
            state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }

            torch.save(state, "checkpoints/metaworld/model_"+str(epoch)+".pt")
            print(f"Model saved!")
            


if __name__=="__main__":
    fname = 'configs/metaworld_data.yaml'
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(params)
    main(params)