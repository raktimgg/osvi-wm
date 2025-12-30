import torch
import numpy as np
import cv2
import random
from hem.datasets.savers.trajectory import Trajectory
import glob
import os
import pickle as pkl
import re

def ema_actions(actions, moving_window_size, tau = 0.01):
    l = actions.shape[1]
    all_ema_actions = []
    for step in range(l):
        start_step = max(0, step-moving_window_size+1)
        end_step = step+1
        actions_for_curr_step = []
        for k in range(end_step-start_step):
            actions_for_curr_step.append(actions[:,end_step-k-1,k])
        actions_for_curr_step = torch.stack(actions_for_curr_step, dim=1)
        # print(actions_for_curr_step.shape)
        exp_weights = np.exp(-tau * np.arange(actions_for_curr_step.shape[1]))
        # print(exp_weights)
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).to(actions_for_curr_step.device).unsqueeze(dim=1).unsqueeze(0)
        exp_weights = exp_weights.expand(actions_for_curr_step.shape[0], -1, -1)
        # print(exp_weights.shape)
        ema_actions = (actions_for_curr_step * exp_weights).sum(dim=1)
        all_ema_actions.append(ema_actions)

    all_ema_actions = torch.stack(all_ema_actions, dim=1)
    return all_ema_actions


SHUFFLE_RNG = 2843014334
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
SAWYER_DEMO_PRIOR = np.array([-1.95254033,  -3.25514605,  1.0,        -2.85691298,  -1.41135844,
                            -1.33966008,  -3.25514405,  1.0,        -2.89548721,  -1.26587143,
                            0.86143858,  -2.36652955,  1.0,        -2.61823206,   0.2176199,
                            -3.54059052,  -4.00911932,  1.0,        -5.07546054,  -5.25952708,
                            -3.7442406,   -5.35087854,  1.0,        -4.2814715,   -3.72755719,
                            -3.85309935,  -5.71775012,  1.0,        -7.02858012,  -3.15234408,
                            -3.67211177, 1.0,         -3.22493535, -12.45458803, -11.76144085, 0, 0])


def resize(image, target_dim, normalize=False, reduce_bits=False):
    if image.shape[:2] != target_dim:
        inter_method = cv2.INTER_AREA
        if np.prod(image.shape[:2]) > np.prod(target_dim):
            inter_method = cv2.INTER_LINEAR
        
        resized = cv2.resize(image, target_dim, interpolation=inter_method)
    else:
        resized = image

    if len(resized.shape) == 2:
        resized = resized[:,:,None]
    
    if reduce_bits:
        assert resized.dtype == np.uint8, "math only works on uint8 data!"
        resized -= resized % 8

    if normalize:
        return (resized.astype(np.float32) / 255 - MEAN) / STD

    return resized.astype(np.float32)


def crop(img, crop):
    if crop[0] > 0:
        img = img[crop[0]:]
    if crop[1] > 0:
        img = img[:-crop[1]]
    if crop[2] > 0:
        img = img[:,crop[2]:]
    if crop[3] > 0:
        img = img[:,:-crop[3]]
    return img


def randomize_video(frames, color_jitter=None, rand_gray=None, rand_crop=None, rand_rot=0, rand_trans=np.array([0,0]), normalize=False, rand_flip=False,force_flip=None):
    frames = [fr.astype(np.float32) for fr in frames]
    
    if color_jitter is not None:
        rand_h, rand_s, rand_v = [np.random.uniform(-h, h) for h in color_jitter]
        delta = np.array([rand_h * 180, rand_s, rand_v]).reshape((1, 1, 3)).astype(np.float32)
        frames = [np.clip(cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_RGB2HSV) + delta, cv2.COLOR_HSV2RGB), 0, 255) for fr in frames]
    if rand_gray and np.random.uniform() < rand_gray:
        frames = [np.tile(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)[:,:,None], (1,1,3)) for fr in frames]

    applied_crop = [0]*4
    if rand_crop is not None:
        r1, r2 = [np.random.randint(rand_crop[0]) for _ in range(2)]
        c1, c2 = [np.random.randint(rand_crop[1]) for _ in range(2)]
        
        h, w = frames[0].shape[:2]
        frames = [fr[r1:] for fr in frames]
        frames = [fr[:-r2] for fr in frames] if r2 else frames
        frames = [fr[:,c1:] for fr in frames]
        frames = [fr[:,:-c2] for fr in frames] if c2 else frames
        frames = [resize(fr, (w, h), normalize=False) for fr in frames]
        applied_crop = [r1,r2,c1,c2]
    trans=np.array([0,0])
    if rand_rot or any(rand_trans):
        rot = np.random.uniform(-rand_rot, rand_rot)
        trans = np.random.uniform(-rand_trans, rand_trans)
        M = np.array([[np.cos(rot), -np.sin(rot), trans[0]], [np.sin(rot), np.cos(rot), trans[1]]])
        frames = [cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0])) for fr in frames]
    if rand_flip:
        if force_flip is not None:
            vert,horz = force_flip
        else:
            vert = -1 if random.random() > 0.5 else 1
            horz = -1 if random.random() > 0.5 else 1
        frames = [x[::vert,::horz] for x in frames]
        flip = np.array([[horz,0,0],[0,vert,0],[0,0,1]])
    else:
        flip = np.eye(3)
    if normalize:
        frames = [(fr.astype(np.float32) / 255 - MEAN) / STD for fr in frames]
    else:
        for fr in frames:
            fr /= 255
    frames = np.concatenate([fr[None] for fr in frames], 0).astype(np.float32)
    return frames, {'crop': applied_crop,'trans':trans,'flip':flip}


# def split_files(file_len, splits, mode='train'):
    # assert sum(splits) == 1 and all([0 <= s for s in splits]), "splits is not valid pdf!"

    # order = [i for i in range(file_len)]
    # random.Random(SHUFFLE_RNG).shuffle(order)
    # pivot = int(len(order) * splits[0])
    # if mode == 'train':
        # order = order[:pivot]
    # else:
        # order = order[pivot:]
    # return order

# splits out tasks
def split_files(file_len, splits, mode='train'):
    assert file_len == 1600
    order = [i for i in range(file_len)]
    if mode == 'all':
        inds = order
    elif mode == 'none':
        inds = []
    elif mode=='train':
        inds = order[:1400]
    else:
        inds = order[1400:]
    random.Random(SHUFFLE_RNG).shuffle(inds)
    return inds

def select_random_frames(frames, n_select, sample_sides=False):
    selected_frames = []
    clip = lambda x : int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    for i in range(n_select):
        n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
        if sample_sides and i == n_select - 1:
            n = len(frames) - 1
        elif sample_sides and i == 0:
            n = 0
        selected_frames.append(n)

    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in selected_frames]
    elif isinstance(frames, Trajectory):
        return [frames[i]['obs']['image'] for i in selected_frames]
    return frames[selected_frames]

def get_files(root_dir):
    def natural_keys(text):
        # source: https://www.tutorialspoint.com/How-to-correctly-sort-a-string-with-a-number-inside-in-Python
        atoi = lambda text: int(text) if text.isdigit() else text
        return [atoi(c) for c in re.split('(\d+)',text)]

    root_dir = os.path.expanduser(root_dir)
    if 'pkl' in root_dir or 'hdf5' in root_dir:
        return sorted(glob.glob(root_dir), key=natural_keys)
    pkl_files = glob.glob(root_dir + '*.pkl')
    hdf5_files = glob.glob(root_dir + '*.hdf5')
    return sorted(pkl_files + hdf5_files, key=natural_keys)

class DummyMujocoPyLoader(pkl.Unpickler):
    def find_class(self, module, name):
        if 'mujoco_py' in module:
            print(f"Skipping mujoco_py class: {module}.{name}")
            return lambda *args, **kwargs: None  # return dummy None
        return super().find_class(module, name)

def load_traj(fname):
    if '.pkl' in fname:
        if 'franka' in fname:
            traj = {}
            traj['obs'] = pkl.load(open(fname, 'rb'))
            traj['setting_name'] = fname.split("/")[-4]
            traj['fname'] = fname
        else:
            if 'high_ent' in fname:
                setting_name = fname.split("/")[-2]
            else:
                setting_name = fname.split("/")[-4]
            traj = pkl.load(open(fname, 'rb'))['traj']
            traj.setting_name = setting_name
            traj.fname = fname
    elif '.hdf5' in fname:
        traj = HDF5Trajectory()
        traj.load(fname)
    else:
        raise NotImplementedError
    import_render_wrapper = False
    traj = traj if not import_render_wrapper else ImageRenderWrapper(traj)
    return traj

# gives files sorted based on first capture group of pattern
def sorted_file_match(directory,pat):
    files = os.listdir(directory)
    matches = [re.match(pat,e) for e in files]
    pairs = [(fil,int(match[1])) for fil,match in zip(files,matches) if match]
    if len(pairs) == 0:
        return [[],[]]
    return list(zip(*sorted(pairs,key=lambda x: x[1])))