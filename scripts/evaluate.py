import numpy as np
import re
import torch
import argparse
from pyutil import *
import torch.utils.data
import argparse
import os
from dblog import DbLog
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm
from scripts.eval_util import test_transformer
import time
import copy
import warnings
from tensorboardX import SummaryWriter
# from metaworld.policies.policy import move

from hem.util import parse_basic_config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path')
    parser.add_argument('--test_task',default=None, choices=['button-press-v2', 'pick-place-wall-v2', 'window-open-v2', 'door-unlock-v2'], help='None for Pick-and-Place')
    parser.add_argument('--instances',default=100,type=int)
    parser.add_argument('--envs',default=40,type=int)
    parser.add_argument('--save_images',default=False,type=bool)

    args = parser.parse_args()
    config_path = args.config_path
    config = parse_basic_config(config_path, resolve_env=True)
    osvi_wm_config = os.path.join(*config_path.split('/')[:-1],config['osvi_wm_config'])
    test_transformer(copy.deepcopy(config),osvi_wm_config=osvi_wm_config,
                    num_waypoints=None,instances=args.instances,
                    test_task=args.test_task,num_envs=args.envs,
                    save_images=args.save_images)
