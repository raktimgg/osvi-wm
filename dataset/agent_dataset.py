from torch.utils.data import Dataset
from einops import repeat
import cv2
import random
import os
import torch
from utils.utils import resize, crop, randomize_video
import random
import numpy as np
import io
import tqdm
from utils.utils import get_files, load_traj
from utils.utils import split_files
import pickle as pkl
from dataset.env_data import ALL_ENVS
from dataset.env_data import TASK_ENV_MAP
from utils.projection_utils import image_point_to_pixels, pixels_to_image_point, embed_mat,embed_matrix_square, compute_crop_adjustment

bcz_correction = np.array([[1./320, 0, -1], [0., -1./256, 1], [0., 0., 1.]])
def traj_to_base_matrix(traj, franka=False):
    if franka:
        if 'human_franka' in traj["setting_name"]:
            # print('loading human franka projection matrix')
            return INVERSE_MATS['human_franka']
        return INVERSE_MATS['franka']
    if 'world_to_image_transform' in traj.get(0)['obs']:
        return np.linalg.inv(traj.get(0)['obs']['world_to_image_transform'])
    if 'cam_mat' in traj.get(0)['obs']:
        return np.linalg.inv(embed_mat(bcz_correction @ traj.get(0)['obs']['cam_mat']))
    if traj.setting_name in ALL_ENVS:
        return INVERSE_MATS['metaworld']
    elif traj.setting_name in TASK_ENV_MAP.keys():
        return INVERSE_MATS[traj.setting_name]
    else:
        return INVERSE_MATS['ost']


def adjust_augmentations(stats,size):
    # account for random crops and translations
    random_crop_adjust = compute_crop_adjustment(stats['crop'],size)
    random_trans_adjust = np.eye(4)
    # 'trans' output is in pixels, [x,y] format, so scale by 2*[c,r] and flip y
    trans = 2*stats['trans']/np.flip(size)
    trans[1] = -trans[1]
    random_trans_adjust[:3,2] = embed_mat(trans,3)
    return np.linalg.inv(random_crop_adjust) @ np.linalg.inv(random_trans_adjust) @embed_mat(stats['flip'])


T_metaworld = np.array([[-0.70710678, -0.19245009, -0.68041382, -1.1       ],
                        [ 0.70710678, -0.19245009, -0.68041382, -0.4       ],
                        [ 0.,         -0.96225045,  0.27216553, 0.6       ],
                        [ 0.,          0.,          0.,          1.        ]])    

T_ost = np.array([[ 1.56091861e-06,  4.24947306e-01,  9.05218088e-01,  1.60000000e+00],
                [-9.99998737e-01, -1.43802503e-03,  6.76793566e-04,  1.00000000e-01],
                [ 1.58932787e-03, -9.05216946e-01,  4.24946767e-01,  1.75000000e+00],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])   


T_franka = np.array([[ 0.03946494,  0.39746716, -0.91676735,  0.96345005],
                    [ 0.9991582,  -0.02597944,  0.03174824,  0.47648504],
                    [-0.01119822, -0.91724856, -0.39815785,  0.54829367],
                    [ 0.,          0.,          0.,          1.        ]]) 

T_human_franka = np.array([[-0.99123969,  0.05960822,  0.11785893,  0.10280887],
                        [-0.04773728,  0.67033809, -0.74051874,  0.74120309],
                        [-0.12314634, -0.73965783, -0.66162018,  0.43353179],
                        [ 0.,          0.,          0.,          1.        ]]) 

INVERSE_MATS = {'metaworld':T_metaworld,'ost':T_ost,'franka':T_franka, 'human_franka':T_human_franka, 'dev':np.eye(4),'.':np.eye(4)}

INVERSE_MATS['basketball'] = np.array([[-3.28832225e-07, -5.86688314e-01,  8.09812831e-01,  6.34713560e-01],
                                [ 1.00000000e+00,  6.29153391e-07,  8.61864792e-07,  6.40763181e-08],
                                [-1.01514249e-06,  8.09812831e-01,  5.86688314e-01,  1.40496778e+00],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
INVERSE_MATS['button'] = np.array([[ 6.04473853e-08, -6.81006562e-01,  7.32277313e-01,  3.91358085e-01],
                            [ 1.00000000e+00,  9.41117926e-08,  4.97538710e-09, -9.55415117e-09],
                            [-7.23042018e-08,  7.32277313e-01,  6.81006562e-01,  1.24658138e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
INVERSE_MATS['door'] = np.array([[-7.26818450e-08, -6.28266450e-01,  7.77998244e-01,  5.98613175e-01],
                                [ 1.00000000e+00,  7.26818450e-08,  1.52115266e-07, -4.39203568e-09],
                                [-1.52115266e-07,  7.77998244e-01,  6.28266450e-01,  1.59035002e+00],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
INVERSE_MATS['drawer'] = np.array([[ 6.04473853e-08, -6.81006562e-01,  7.32277313e-01,  3.91358085e-01],
                                [ 1.00000000e+00,  9.41117926e-08,  4.97538710e-09, -9.55415117e-09],
                                [-7.23042018e-08,  7.32277313e-01,  6.81006562e-01,  1.24658138e+00],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
INVERSE_MATS['nut_assembly'] = np.array([[ 0.,         -0.70614784,  0.70806442,  0.5       ],
                                        [ 1.,          0.,          0.,          0.        ],
                                        [ 0.,          0.70806442,  0.70614784,  1.35      ],
                                        [ 0.,          0.,          0.,          1.        ]])
INVERSE_MATS['pick_place'] = np.array([[ 6.04473853e-08, -6.81006562e-01,  7.32277313e-01,  3.91358085e-01],
                                    [ 1.00000000e+00,  9.41117926e-08,  4.97538710e-09, -9.55415117e-09],
                                    [-7.23042018e-08,  7.32277313e-01,  6.81006562e-01,  1.24658138e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
INVERSE_MATS['stack_block'] = np.array([[ 6.04473853e-08, -6.81006562e-01,  7.32277313e-01,  3.91358085e-01],
                                        [ 1.00000000e+00,  9.41117926e-08,  4.97538710e-09, -9.55415117e-09],
                                        [-7.23042018e-08,  7.32277313e-01,  6.81006562e-01,  1.24658138e+00],
                                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


class AgentDemonstrations(Dataset):
    def __init__(self, root_dir=None, files=None, height=224, width=224, depth=False, normalize=True, crop=None, randomize_vid_frames=False, T_context=15, extra_samp_bound=0,
                 T_pair=0, freq=1, append_s0=False, mode='train', split=[0.9, 0.1], state_spec=None, action_spec=None, sample_sides=False, min_frame=0, cache=False, random_targets=False,
                 color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, rep_buffer=0, target_vid=False, reduce_bits=False, aux_pose=False,waypoints=False,
                 no_context_jitter=False,grasp=True,rand_flip=False,interpolate_gaps=False,high_ent=False,head_label=0,raw_images=False,start_samp_rate=0,hand_cam=False, franka=False):
        assert mode in ['train', 'val'], "mode should be train or val!"
        assert T_context >= 2 or T_pair > 0, "Must return (s,a) pairs or context!"

        self.franka = franka
        self.raw_images = raw_images
        self.hand_cam = hand_cam
        self._rand_flip = rand_flip
        self.waypoints = waypoints
        self.no_context_jitter = no_context_jitter
        self.interpolate_gaps = interpolate_gaps
        self.high_ent = high_ent
        self.head_label=head_label
        self.start_samp_rate = start_samp_rate
        if high_ent and head_label == 0:
            raise Exception(f'bad label')
        if files is None and rep_buffer:
            all_files = []
            for f in range(rep_buffer):
                all_files.extend(pkl.load(open(os.path.expanduser(root_dir.format(f)), 'rb')))
            order = split_files(len(all_files), split, mode)
            files = [all_files[o] for o in order]
        elif files is None:
            all_files = get_files(root_dir)
            order = split_files(len(all_files), split, mode)
            files = [all_files[o] for o in order]
        self._trajs = files
        if cache:
            for i in tqdm.tqdm(range(len(self._trajs))):
                if isinstance(self._trajs[i], str):
                    self._trajs[i] = load_traj(self._trajs[i])

        self._im_dims = (width, height)
        self._randomize_vid_frames = randomize_vid_frames
        self._crop = tuple(crop) if crop is not None else (0, 0, 0, 0)
        self._depth = depth
        self._normalize = normalize
        self._T_context = T_context
        self._T_pair = T_pair
        self._freq = freq
        state_spec = tuple(state_spec) if state_spec is not None else ('ee_aa', 'ee_vel', 'joint_pos', 'joint_vel', 'gripper_qpos', 'object_detected')
        action_spec = tuple(action_spec) if action_spec is not None else ('action',)
        self._state_action_spec = (state_spec, action_spec)
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_rot = rand_rotate if rand_rotate is not None else 0
        if not is_rad:
            self._rand_rot = np.radians(self._rand_rot)
        self._rand_trans = np.array(rand_translate if rand_translate is not None else [0, 0])
        self._rand_gray = rand_gray
        self._normalize = normalize
        self._append_s0 = append_s0
        self._sample_sides = sample_sides
        self._target_vid = target_vid
        self._reduce_bits = reduce_bits
        self._min_frame = min_frame
        self._extra_samp_bound = extra_samp_bound
        self._random_targets = random_targets
        self._aux_pose = aux_pose
        self.grasp = grasp

    def __len__(self):
        return len(self._trajs)
    
    def __getitem__(self, index,force_flip = None):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self._trajs), "invalid index!"
        return self.proc_traj(self.get_traj(index),force_flip)
    
    def get_traj(self, index):
        if isinstance(self._trajs[index], str):
            return load_traj(self._trajs[index])
        return self._trajs[index]

    def proc_traj(self, traj,force_flip=None):
        context_frames = []
        if self._T_context:
            context_frames = self._make_context(traj,force_flip=force_flip)

        if self._T_pair == 0:
            return {}, context_frames
        return self._get_pairs(traj,force_flip=force_flip), context_frames

    def _make_context(self, traj,force_flip=None):
        if self.franka:
            len_traj = len(traj['obs']['images'])
        else:
            len_traj = len(traj)
        clip = lambda x : int(max(0, min(x, len_traj - 1)))
        per_bracket = max(len_traj / self._T_context, 1)
        def _make_frame(n):
            if not self.franka:
                obs = traj.get(n)['obs']
                img = self._crop_and_resize(obs['image'])
            else:
                obs = traj['obs']
                img = self._crop_and_resize(obs['images'][n][:,:,:3].astype(np.uint8))
            if self._depth:
                img = np.concatenate((img, self._crop_and_resize(obs['depth'][:,:,None])), -1)
            return img[None]

        frames = []
        if self.no_context_jitter:
            teacher_im_inds = np.linspace(0,len_traj-1,num=self._T_context,endpoint=True,dtype=int)
            frames = [_make_frame(i) for i in teacher_im_inds]
        else:
            for i in range(self._T_context):
                n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
                if self._sample_sides and i == self._T_context - 1:
                    n = len_traj - 1
                elif self._sample_sides and i == 0:
                    n = self._min_frame
                frames.append(_make_frame(n))
        frames = np.concatenate(frames, 0)
        frames,stats = randomize_video(frames, self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize,rand_flip=self._rand_flip,force_flip=force_flip)
        projection = traj_to_base_matrix(traj, self.franka)  # the project matrix for franka here is for the agent and not the teacher, but this doesn't matter as this is not used elsewhere
        size = frames.shape[-3:-1]
        crop_adjust = compute_crop_adjustment(self._crop, size)
        projection = projection @ np.linalg.inv(crop_adjust)
        projection = projection @ adjust_augmentations(stats,size)
        if not self.franka:
            return {'video': np.transpose(frames, (0, 3, 1, 2)), 'projection_matrix': projection,'fname':traj.fname}
        else:
            return {'video': np.transpose(frames, (0, 3, 1, 2)), 'projection_matrix': projection,'fname':traj['fname']}

    def _get_pairs(self, traj, end=None,force_flip=None):
        def _get_tensor(k, t):
            if k == 'action':
                return t['action']
            elif k == 'grip_action':
                return [t['action'][-1]]

            o = t['obs']
            if k == 'ee_aa' and 'ee_aa' not in o:
                ee, axis_angle = o['ee_pos'][:3], o['axis_angle']
                if axis_angle[0] < 0:
                    axis_angle[0] += 2
                o = np.concatenate((ee, axis_angle)).astype(np.float32)
            else:
                o = o[k]
            return o
        
        state_keys, action_keys = self._state_action_spec
        ret_dict = {'images': []}
        traj_elements = [x for x in traj]

        end = len(traj) if end is None else end
        if self.waypoints:
            start = 0
        
        
        self._freq = 10
        if self.franka:
            len_traj = len(traj['obs']['images'])
        else:
            len_traj = len(traj)
        # start = np.random.randint(0, max(1,len(traj) - self._T_pair*self._freq - 1))
        # chosen_t = [min(j * self._freq + start, len(traj)-1) for j in range(self._T_pair + 1)]
        chosen_t = np.linspace(start,len_traj-1,num=self._T_pair+1,endpoint=True,dtype=int)
        # print(chosen_t)
        # if self._T_pair > 1:
        #     chosen_t = [0] + np.linspace(1,len(traj)-1,num=self._T_pair-1,endpoint=True,dtype=int).tolist()

        for j, t in enumerate(chosen_t):
            if not self.franka:
                if len(traj) < 2:
                    print(traj.fname, t, len(traj))
                    assert False, traj.fname
                t = traj_elements[t]
                image = t['obs']['image']
            else:
                image = traj['obs']['images'][t][:,:,:3].astype(np.uint8)
            ret_dict['images'].append(self._crop_and_resize(image)[None])
            
        for k, v in ret_dict.items():
            ret_dict[k] = np.concatenate(v, 0).astype(np.float32)

        ret_dict['images'],randomize_stats = self.randomize_frames(ret_dict['images'],ret_stats=True,force_flip=force_flip)
        ret_dict['images'] = np.transpose(ret_dict['images'], (0, 3, 1, 2))

        # out_inds = np.linspace(0,len(traj)-1,num=50,endpoint=True,dtype=int)#type: ignore

        ret_dict['setting_name'] = traj.setting_name if not self.franka else traj['setting_name']
        if self.franka:
            # for franka env
            grasp_frames = [traj['obs']['grasp'][i]<0.5 for i in range(0,len_traj)]  # in franka traj, 1 is gripper open, 0 is gripper close
        elif 'grasp' in traj_elements[0]['obs']:
            # for metaworld/bcz/mosaic
            grasp_frames = [traj_elements[i]['obs']['grasp'] for i in range(0,len(traj))]
        else:
            # for pick place env
            grasp_frames = [False] + [traj_elements[i]['action'][-1] > 0.01 for i in range(1,len(traj))]

        # out_inds = np.linspace(0,len(traj)-1,num=50,endpoint=True,dtype=int)#type: ignore
        out_inds = np.linspace(start,len_traj-1,num=50,endpoint=True,dtype=int)#type: ignore
        if not self.franka:
            poses = np.stack([traj_elements[i]['obs']['ee_aa'][:3] for i in out_inds])
        else:
            poses = np.stack([traj['obs']['ee_aa'][i][:3] for i in out_inds])
        grasps = np.stack([grasp_frames[i] for i in out_inds]).astype(np.int32)
        # grasp scale is 0.2 this is tunable
        traj_points = np.concatenate((poses,grasps[:,None]*0.2),axis=-1)
        ret_dict['traj_points'] = traj_points
        ret_dict['start0'] = start == 0
        grasp_ind = np.where(grasps)[0]

        ret_dict['projection_matrix'] = traj_to_base_matrix(traj, self.franka)
        size = ret_dict['images'].shape[-2:]
        crop_adjust = compute_crop_adjustment(self._crop, size)
        ret_dict['projection_matrix'] = ret_dict['projection_matrix'] @ np.linalg.inv(crop_adjust)
        ret_dict['projection_matrix'] = ret_dict['projection_matrix'] @ adjust_augmentations(randomize_stats[0],size)
        ret_dict['high_ent'] = int(self.high_ent)
        ret_dict['head_label'] = int(self.head_label)
        return ret_dict

    def randomize_frames(self,frames,ret_stats = False,force_flip=None):
        if self._randomize_vid_frames:
            result = [randomize_video([f], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize,rand_flip = self._rand_flip,force_flip=force_flip) for f in frames]
            result,stats = zip(*result)
            result = np.concatenate(result, 0)
        else:
            result,stats = randomize_video(frames, self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize,rand_flip = self._rand_flip,force_flip=force_flip)
            stats = [stats]
        if ret_stats:
            return result,stats
        else:
            return result
    
    def _crop_and_resize(self, img, normalize=False):
        return resize(crop(img, self._crop), self._im_dims, normalize, self._reduce_bits)
    
    def _adjust_points(self, points, frame_dims):
        h = np.clip(points[0] - self._crop[0], 0, frame_dims[0] - self._crop[1])
        w = np.clip(points[1] - self._crop[2], 0, frame_dims[1] - self._crop[3])
        h = float(h) / (frame_dims[0] - self._crop[0] - self._crop[1]) * self._im_dims[1]
        w = float(w) / (frame_dims[1] - self._crop[2] - self._crop[3]) * self._im_dims[0]
        return tuple([int(min(x, d - 1)) for x, d in zip([h, w], self._im_dims[::-1])])


if __name__ == '__main__':
    import time
    import imageio
    from torch.utils.data import DataLoader
    batch_size = 10
    ag = AgentDemonstrations('/dev/shm/mc48/metaworld_traj_eef_depth', normalize=False)
    loader = DataLoader(ag, batch_size = batch_size, num_workers=8)

    start = time.time()
    timings = []
    for pairs, context in loader:
        timings.append(time.time() - start)
        print(context.shape)

        if len(timings) > 1:
            break
        start = time.time()
    print('avg ex time', sum(timings) / len(timings) / batch_size)

    out = imageio.get_writer('out1.gif')
    for t in range(context.shape[1]):
        frame = [np.transpose(fr, (1, 2, 0)) for fr in context[:, t]]
        frame = np.concatenate(frame, 1)
        out.append_data(frame.astype(np.uint8))
    out.close()
