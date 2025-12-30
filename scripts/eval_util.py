import numpy as np
import numpy as np
import numpy as np
import numpy as np
import yaml
import re
import torch
import argparse
from envs.gripper_env import GripperEnv
from pyutil import *
import torch.utils.data
import argparse
import os
from dblog import DbLog
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm
from einops import rearrange, repeat
import functools
import pickle
import time
from hem.models.inverse_module import InverseImitation
from models.model import StateSpaceModel
from hem.datasets import get_dataset
from envs.metaworld_env import make_metaworld_single, ALL_ENVS
from hem.datasets.teacher_dataset import TeacherDemonstrations
from hem.datasets.agent_dataset import AgentDemonstrations
from hem.datasets.agent_teacher_dataset import AgentTeacherDataset
from policies.waypoints_policy import WaypointsPolicy
from utils.projection_utils import image_coords_to_3d
from utils.projection_utils import image_point_to_pixels, pixels_to_image_point, embed_mat,embed_matrix_square, compute_crop_adjustment
from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory
import cv2
import pathlib

def sample_cont(dcfg,teacher_motion):
    dcfg = dcfg.copy()
    dcfg['test_tasks'] = [teacher_motion]
    dataset = AgentTeacherDataset(**dcfg,mode='test')
    ind = random.choice(np.arange(len(dataset)))
    if dcfg.get('mosaic'):
        task_string = pathlib.Path(dataset[ind][0]['fname']).parts[-2]
        subtask = int(re.match('task_(\d+)',task_string)[1])
    else:
        subtask = None
    return torch.tensor(dataset[ind][0]['video']), subtask

def test_transformer_all(config,**kwargs):
    _,nums = sorted_file_match(f"{config.logdir}/models",r'model_(\d+).torch')
    for n in nums:
        test_transformer(config,n,**kwargs)

def make_metaworld_env(task,cfg=None,seed=0):
    # not the most robust way to get the right depth
    # depth = 'depth' in cfg['dataset']['agent_dir']
    depth = False
    env = make_metaworld_single(task,random_init=True,seed=seed,rgb=(not depth),depth_only=depth,object_state=False,handcam=True)
    # force framestack 1 for now
    env = DictFrameStack(env,1)
    return env

def make_env(task,cfg=None,sub_task=None,seed=None):
    if seed is None:
        seed = random.randint(0,100000)
    if task in ALL_ENVS:
        return make_metaworld_env(task,cfg,seed)
    elif task == 'ost' or task in list(range(16)):
        return DictFrameStack(GripperEnv(task,rgb=True,handcam=True,seed=seed),1)
        # return get_expert_trajectory('PandaPickPlaceDistractor', task=sub_task, ret_env=True, seed=seed,hand_depth=True)
    else:
        return DictFrameStack(make_mosaic_env(task,sawyer=False,task=sub_task,handcam=True),1)

import pickle as pkl
from hem.datasets.agent_dataset import INVERSE_MATS
# PROJECTION_MATRICES = pkl.load(open('utils/transformation_mats_square.pkl','rb'))
# INVERSE_MATS = {key:np.linalg.inv(mat)for key,mat in PROJECTION_MATRICES.items()}
from envs.metaworld_env import ALL_ENVS
def get_pix_to_3d_projection(mot):
    if mot in ALL_ENVS:
        return INVERSE_MATS['metaworld']
    elif mot in list(range(16)):
        # crop_adjust = compute_crop_adjustment([100,0,0,0], [240,320])
        # return INVERSE_MATS['ost'] @ np.linalg.inv(crop_adjust)
        return INVERSE_MATS['ost']
    else:
        return INVERSE_MATS[mot] 

def get_model(config_file):
    with open(config_file, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)
    model_config = args["model"]
    data_config = args["data"]
    metaworld = data_config.get('metaworld', False)

    latent_dim = model_config["latent_dim"]
    waypoints = model_config["waypoints"]
    sub_waypoints = model_config["sub_waypoints"]
    model = StateSpaceModel(latent_dim=latent_dim, waypoints=waypoints, sub_waypoints=sub_waypoints, metaworld=metaworld)
    osvi_wm_base_loc = os.path.join(*config_file.split('/')[:-2])
    if metaworld:
        checkpoint_path = f'{osvi_wm_base_loc}/checkpoints/metaworld/model.pt'
    else:
        checkpoint_path = f'{osvi_wm_base_loc}/checkpoints/pp/model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
    pretrained_dict = checkpoint
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    print("Loaded pretrained weights for the model.") 
    model = model.eval().cuda()
    return model, metaworld

def test_transformer(config,osvi_wm_config,num_waypoints=None,instances=10,test_task='button-press-v2',num_envs=40,save_images=False):
    if num_waypoints is None:
        num_waypoints = config['policy'].get('waypoints',0)

    model, metaworld = get_model(osvi_wm_config)
    
    dataset_config = config['dataset']
    if metaworld:
        dataset_config['test_tasks'] = [test_task]

    dataset_class = get_dataset(dataset_config.pop('type'))
    dataset = dataset_class(**dataset_config, mode='test')
    motions = dataset.test_tasks
    all_mots = repeat(np.array(motions),'n -> n b', b = instances).ravel()
    splits = np.array_split(all_mots,math.ceil(len(all_mots)/num_envs))
    mosaic = dataset_config.get('mosaic',False)
    is_ost = not mosaic and not metaworld
    all_successes = 0
    for idx, mots in enumerate(splits):
        print('Evaluating part', idx+1,'of',len(splits),'Number of tasks',len(mots))
        contexts,subtasks = zip(*[sample_cont(dataset_config,mot) for mot in mots])
        contexts = torch.stack(contexts).cuda()
        envs = SubprocVecEnv([functools.partial(make_env,m,config,st) for m,st in zip(mots,subtasks)],'spawn')
        projection_mats = torch.tensor(np.stack([get_pix_to_3d_projection(mot) for mot in mots])).cuda().float()
        
        # ROLLOUTS
        states = []
        ims = []
        actions = []
        rewards = []
        infos = []
        masks = []
        state = envs.reset()
        # compute waypoints
        states.append(state)

        start_im = state['img'][...,:3]
        resized = np.stack([dataset._agent_dataset._crop_and_resize(x[0]) for x in start_im])
        start_im,transform_stats = dataset._agent_dataset.randomize_frames(resized,ret_stats=True)

        im_shape = start_im.shape[-3:-1]
        crop_adjust = compute_crop_adjustment(dataset._agent_dataset._crop, im_shape)
        trans_mat = torch.tensor(np.linalg.inv(crop_adjust)).cuda().float()
        projection_mats = torch.stack([mat @ trans_mat for mat in projection_mats])

        start_im = to_channel_first(torch.tensor(start_im[:,None]))
        start_im = start_im.cuda()
        start_im = repeat(start_im,'b 1 c r col->b 2 c r col')
        policies = []
        if config.get('waypoints',False):
            with torch.no_grad():
                start_poses = state['state'][:,-1,:4] #type: ignore
                flatstate = rearrange(state['state'],'b t d -> b (t d)') #type: ignore
                head_label = dataset_config.get('head_label',0)
                head_label = torch.ones((start_im.shape[0],)).cuda()*head_label

                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(start_im, contexts, T_tot=16)
                
                if config.get('image_waypoints'):
                    waypoints = image_coords_to_3d(out['waypoints'],projection_mats)
                    waypoints[...,3:] = out['waypoints'][...,3:]
                    world_waypoints = waypoints.cpu().numpy()
                else:
                    waypoints = out['waypoints'].cpu().numpy()
                    shiftby = start_poses[:,None,:]
                    shiftby[:,:,-1] = 0
                    world_waypoints = waypoints + shiftby

                if config['policy'].get('sub_waypoints',False):
                    start = num_waypoints*(num_waypoints-1)//2
                else:
                    start = 0
                inter_waypoints = world_waypoints[:,start:start+num_waypoints]
                for i,wp in enumerate(inter_waypoints):
                    envs.env_method('set_waypoints',wp,indices=[i])
                policies = [WaypointsPolicy(x,panda=(is_ost or mosaic),mosaic=mosaic,verbose=False) for x in inter_waypoints]
                
        done = None
        info = None
        images = []
        hand_cam_images = []
        image_list = []
        finished = None
        kk = 0
        start_time = time.time()
        do_replan = (metaworld and (test_task=='window-open-v2' or test_task=='door-unlock-v2'))
        if do_replan:
            max_steps = 2000
        else:
            max_steps = 500
        print('Do Replan:', do_replan)
        while finished is None or not finished.all():
            if do_replan and ((kk+1)%500==0):
                start_im = torch.Tensor(state['img'][...,:3]).repeat(1,2,1,1,1).permute(0,1,4,2,3)/255.
                MEAN = torch.Tensor([0.485, 0.456, 0.406]).view(1,1,3,1,1).to(start_im.device)
                STD = torch.Tensor([0.229, 0.224, 0.225]).view(1,1,3,1,1).to(start_im.device)
                start_im = (start_im - MEAN) / STD
                with torch.inference_mode():
                    start_im = start_im.to(contexts.device)
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        out = model(start_im, contexts, T_tot=16)
                    waypoints = image_coords_to_3d(out['waypoints'],projection_mats)
                    waypoints[...,3:] = out['waypoints'][...,3:]
                world_waypoints = waypoints.cpu().numpy()
                start = num_waypoints*(num_waypoints-1)//2
                inter_waypoints = world_waypoints[:,start:start+num_waypoints]
                for i,wp in enumerate(inter_waypoints):
                    envs.env_method('set_waypoints',wp,indices=[i])
                policies = [WaypointsPolicy(x,panda=(is_ost or mosaic),mosaic=mosaic,verbose=False) for x in inter_waypoints]
            
            actions = []
            for st,im,policy in zip(state['state'],state['img'],policies): #type: ignore
                assert st.shape[0] == 1 and im.shape[0] == 1
                action = policy.act({'state':st[0],'img':im[0,:,:,-1]})[0]
                actions.append(action)
            action = np.stack(actions)
         
            state,reward,done,info = envs.step(action)
            done = []
            for i in range(len(mots)):
                done.append(np.array(info[i]['is_success']))    # querying success from info dictionary
            done = np.stack(done)
            # When an episode first returns success, record that success but allow the episode to continue running
            # in the background until either all the other episodes succeed or max_steps is reached. Since multiple episodes
            # run in parallel, we terminate all episodes simultaneously once either all are successful or max_steps has been reached.
            finished = np.logical_or(finished,done) if finished is not None else done

            temp_imgs = np.array([x[0] for x in state['img'][..., :3]])
            if kk%1==0 and save_images:
                for ii in range(temp_imgs.shape[0]):
                    cv2.imwrite(f'photos/{mots[ii]}_{ii}_{kk}.png',temp_imgs[ii][:,:,::-1])
            kk += 1
            print(f"\r Timesteps: {kk}, Successes: {int(np.sum(finished * 1.0))}/{len(mots)}, time={time.time()-start_time:.2f} seconds", end="", flush=True)

            if kk%100==0:
                image_list.append(temp_imgs)
            if kk >= max_steps:
                break
        all_successes += np.sum(finished*1.0)
        print(' ')
    print('Success Rate', all_successes/len(all_mots))
    return

def states_to_poses(states):
    if len(states.shape) == 3:
        return states[...,-1,:3]
    else:
        return states[...,:3]

def waypoints_to_actions(states,waypoints):
    poses = states_to_poses(states)
    diffs = waypoints - poses 
    actions = np.clip(diffs*10,-1,1)
    grips = np.ones((states.shape[0],1))*-1
    return np.concatenate((actions,grips),axis=-1)

