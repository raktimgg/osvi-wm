from torch.utils.data import Dataset
from dataset.agent_dataset import AgentDemonstrations
from dataset.teacher_dataset import TeacherDemonstrations
from utils.utils import get_files, load_traj
import torch
import os
import numpy as np
import json
from utils.utils import split_files
from utils.utils import sorted_file_match
import itertools
import random
import glob

def load_ost(agent_dir,teacher_dir,split,mode,traj_per_task=100,ost_all=False,inds=None):
    pairs = []
    agent_files = get_files(agent_dir)
    teacher_files = get_files(teacher_dir)
    print(f"LOADING: {agent_dir}")
    assert len(agent_files) == len(teacher_files)
    if ost_all:
        if mode == 'train':
            mode = 'all'
        else:
            mode = 'none'
    if inds:
        order = inds
    else:
        order = split_files(len(agent_files), split, mode)
    file_2_o = {order[i]:i for i in range(len(order))}
    for i in range(len(order)):
        traj_ind = int(order[i] // traj_per_task)
        for t in range(traj_per_task):
            t = t + traj_ind * traj_per_task
            if t in file_2_o:
                pairs.append((i, file_2_o[t]))
    # check that things are the same task 
    for p1,p2 in pairs:
        if order[p1]//traj_per_task != order[p2]//traj_per_task or abs(order[p1]-order[p2]) > traj_per_task:
            raise Exception('bad match')
    return pairs,order,agent_files,teacher_files

class AgentTeacherDataset(Dataset):
    def __init__(self, agent_dir, teacher_dir, agent_context=0, traj_per_task=1, epoch_repeat=1, mode='train', 
                 split=[0.9, 0.1], metaworld=False,mosaic=False,bcz=False,franka=False,add_metaworld=None,train_tasks=[],
                 test_tasks=[],ost_all=False,novar=False,flip_sync=False,high_ent=False,epoch_repeat_train=None,
                 agent_name=None,teacher_name=None,**params):
        teacher_context = params.pop('T_context', 15)
        self._agent_context = agent_context = agent_context if agent_context is not None else teacher_context
        self.mosaic = mosaic
        self.agent_dir = agent_dir
        self.metaworld = metaworld
        self.franka = franka
        self.bcz = bcz
        self.add_metaworld = add_metaworld
        self._pairs = []
        self.novar = novar
        self.flip_sync = flip_sync
        self.high_ent=high_ent
        if self.high_ent:
            self.test_tasks = sorted(test_tasks)
            all_tasks = sorted(os.listdir(agent_dir))
            self.train_tasks = sorted(list(set(all_tasks) - set(self.test_tasks)))
            tasks = self.train_tasks if mode == 'train' else []
            agent_files= []
            teacher_files=[]
            for task in tasks:
                trajs = sorted(glob.glob(os.path.join(agent_dir,task,'*.pkl')))
                agent_inds = np.arange(len(trajs)) + len(agent_files)
                teacher_inds = np.arange(len(trajs)) + len(teacher_files)
                agent_files += trajs
                teacher_files += trajs
                self._pairs += list(zip(agent_inds, teacher_inds))
            order = list(range(len(teacher_files)))
            order_t,order_a = order,order
        elif self.mosaic or metaworld or bcz:
            to_load = []
            if self.mosaic: to_load.append(('mosaic',agent_dir))
            if self.bcz: to_load.append(('bcz',agent_dir))
            if self.metaworld: to_load.append(('metaworld',agent_dir))
            if self.add_metaworld is not None and mode == 'train': to_load.insert(0,('add_metaworld',add_metaworld))
            teacher_files = []
            agent_files = []
            self.train_tasks = train_tasks
            self.test_tasks = test_tasks
            if metaworld or bcz or mosaic:
                all_tasks = sorted(os.listdir(agent_dir))
                self.train_tasks = sorted(list(set(all_tasks) - set(self.test_tasks)))
            for load_type,load_dir in to_load:
                if mode == 'train':
                    if load_type == 'add_metaworld':
                        tasks = sorted(os.listdir(load_dir))
                    else:
                        tasks = self.train_tasks
                else:
                    tasks = self.test_tasks
                for task in tasks:
                    if teacher_name is None or agent_name is None:
                        if load_type == 'metaworld':
                            teacher_name,agent_name = 'gripper','gripper'
                        elif load_type == 'bcz':
                            teacher_name,agent_name = 'robot','robot'
                    if load_type == 'mosaic':
                        teacher_name = f'sawyer_{task}'
                        agent_name = f'panda_{task}'
                    names,_ = sorted_file_match(os.path.join(load_dir,task,teacher_name),r'task_(\d+)')
                    for name in names:
                        agent_folder = os.path.join(load_dir,task,agent_name,name)
                        task_agent_files = [os.path.join(agent_folder,x) for x in sorted(os.listdir(agent_folder))]
                        teacher_folder = os.path.join(load_dir,task,teacher_name,name)
                        task_teacher_files = [os.path.join(teacher_folder,x) for x in sorted(os.listdir(teacher_folder))]
                        agent_inds = np.arange(len(task_agent_files)) + len(agent_files)
                        teacher_inds = np.arange(len(task_teacher_files)) + len(teacher_files)
                        agent_files += task_agent_files
                        teacher_files += task_teacher_files
                        if self.novar:
                            self._pairs += list(zip(agent_inds, teacher_inds))
                        else:
                            self._pairs += list(itertools.product(agent_inds,teacher_inds))
            order_t = list(range(len(teacher_files)))
            order_a = list(range(len(agent_files)))
        elif self.franka:
            teacher_files = []
            agent_files = []
            all_tasks = sorted(os.listdir(agent_dir))
            train_tasks = sorted(list(set(all_tasks) - set(test_tasks)))
            self.train_tasks = train_tasks
            self.test_tasks = test_tasks
            if mode == 'train':
                tasks = train_tasks
            else:
                tasks = test_tasks
            for task in tasks:
                agent_folder = os.path.join(agent_dir,task)
                task_agent_files = [os.path.join(agent_folder,x) for x in sorted(os.listdir(agent_folder))]
                teacher_folder = os.path.join(teacher_dir,task)
                task_teacher_files = [os.path.join(teacher_folder,x) for x in sorted(os.listdir(teacher_folder))]
                agent_inds = np.arange(len(task_agent_files)) + len(agent_files)
                teacher_inds = np.arange(len(task_teacher_files)) + len(teacher_files)
                agent_files += task_agent_files
                teacher_files += task_teacher_files
                if self.novar:
                    self._pairs += list(zip(agent_inds, teacher_inds))
                else:
                    self._pairs += list(itertools.product(agent_inds,teacher_inds))
            order_t = list(range(len(teacher_files)))
            order_a = list(range(len(agent_files)))
        else:
            if len(test_tasks) == 0 and len(train_tasks) == 0:
                self.train_tasks = list(range(0,14))
                self.test_tasks = list(range(14,16))
                inds = None
            elif ost_all:
                self.train_tasks = list(range(0,16))
                self.test_tasks = []
                inds = None
            else:
                self.test_tasks = test_tasks
                if len(train_tasks) > 0:
                    self.train_tasks = train_tasks
                else:
                    self.train_tasks = sorted(list(set(range(16)) - set(test_tasks)))
                tasks = self.train_tasks if mode == 'train' else self.test_tasks
                inds = sum([list(range(t*traj_per_task,t*traj_per_task+traj_per_task)) for t in tasks],[])
            self._pairs,order,agent_files,teacher_files = load_ost(agent_dir,teacher_dir,split,mode,traj_per_task,ost_all,inds=inds)
            order_t,order_a = order,order
        if mode != 'train':
            params = params.copy()
            params['rand_translate'] = None
            params['color_jitter'] = None
            if params['rand_crop'] is not None:
                cr = params['rand_crop'][0]//2
                cc = params['rand_crop'][1]//2
                params['crop'] = np.array(params['crop']) + [cr,cr,cc,cc]
            params['rand_flip'] = None
            params['rand_crop'] = None
        params['franka'] = self.franka
        self._agent_dataset = AgentDemonstrations(files=[agent_files[o] for o in order_a], T_context=agent_context, high_ent=high_ent,**params)
        self._teacher_dataset = TeacherDemonstrations(files=[teacher_files[o] for o in order_t], T_context=teacher_context, high_ent=high_ent,**params)
        self._epoch_repeat = epoch_repeat
        if epoch_repeat_train is not None and mode == 'train':
            self._epoch_repeat = epoch_repeat_train

    def __len__(self):
        return len(self._pairs) * self._epoch_repeat
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"
        
        a_i, t_i = self._pairs[index % len(self._pairs)]
        label = index // 10000
        # print(a_i, t_i)
        if self.flip_sync:
            vert = -1 if random.random() > 0.5 else 1
            horz = -1 if random.random() > 0.5 else 1
            force_flip = [vert,horz]
            agent_pairs, agent_context = self._agent_dataset.__getitem__(a_i,force_flip)
            teacher_context = self._teacher_dataset.__getitem__(t_i,force_flip)
        else:
            agent_pairs, agent_context = self._agent_dataset[a_i]
            teacher_context = self._teacher_dataset[t_i]
        
        if self._agent_context:
            return teacher_context, agent_context, agent_pairs
        return teacher_context, agent_pairs, label


class PairedAgentTeacherDataset(Dataset):
    def __init__(self, root_dir, mode='train', split=[0.9, 0.1], **params):
        self._root_dir = os.path.expanduser(root_dir)
        with open(os.path.join(self._root_dir, 'mappings_1_to_1.json'), 'r') as f:
            self._mapping = json.load(f)
        self._teacher_files = sorted(list(self._mapping.keys()))
        self._teacher_files = [self._teacher_files[o] for o in split_files(len(self._teacher_files), split, mode)]

        self._agent_dataset = AgentDemonstrations(files=[], **params)
        self._teacher_dataset = TeacherDemonstrations(files=[], **params)

    def __len__(self):
        return len(self._teacher_files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        teacher_traj = load_traj(os.path.join(self._root_dir, self._teacher_files[index]))
        agent_traj = load_traj(os.path.join(self._root_dir, self._mapping[self._teacher_files[index]]))
        _, agent_context = self._agent_dataset.proc_traj(agent_traj)
        teacher_context = self._teacher_dataset.proc_traj(teacher_traj)
        return teacher_context, agent_context


class LabeledAgentTeacherDataset(PairedAgentTeacherDataset):
    def __init__(self, root_dir, ignore_actor=False, **params):
        self._agent_dataset = AgentDemonstrations(os.path.join(root_dir, 'traj*_robot'), **params)
        self._teacher_dataset = TeacherDemonstrations(os.path.join(root_dir, 'traj*_human'), **params)
        self._ignore_actor = ignore_actor

    def __len__(self):
        if self._ignore_actor == 'agent':
            return len(self._teacher_dataset)
        if self._ignore_actor == 'teacher':
            return len(self._agent_dataset)
        return len(self._agent_dataset) + len(self._teacher_dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        if self._ignore_actor == 'agent':
            ctx1, ctx2 = [self._teacher_dataset[index] for _ in range(2)]
        elif self._ignore_actor == 'teacher':
            ctx1, ctx2  = [self._agent_dataset[index][1] for _ in range(2)]
        else:
            if index < len(self._agent_dataset):
                ctx1, ctx2  = [self._agent_dataset[index][1] for _ in range(2)]
            else:
                ctx1, ctx2  = [self._teacher_dataset[index - len(self._agent_dataset)] for _ in range(2)]
        
        return ctx1, int(index < len(self._agent_dataset)), ctx2
