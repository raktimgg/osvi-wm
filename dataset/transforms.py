import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms

import dataset.video.transforms as video_transforms
from dataset.video.randerase import RandomErasing


def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    normalize=((0.485, 0.456, 0.406),
               (0.229, 0.224, 0.225))
):

    _frames_augmentation = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=normalize,
    )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)
        # if not self.auto_augment:
        #     # Without auto-augment, PIL and tensor conversions simply scale uint8 space by 255.
        #     self.mean *= 255.
        #     self.std *= 255.

        # check if crop size is int or list
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=crop_size,
            auto_augment='rand-m1-n3-mstd0.5-inc1',
            interpolation='bicubic',
        )

        self.spatial_transform = video_transforms.random_resized_crop_with_shift \
            if motion_shift else video_transforms.random_resized_crop

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):
        if self.auto_augment:
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.autoaug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # T C H W
            buffer = buffer.permute(0, 2, 3, 1)  # T H W C
        else:
            buffer = torch.tensor(buffer, dtype=torch.float32)/255.0

        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        if isinstance(self.crop_size, int):
            height, width = self.crop_size, self.crop_size
        else:
            height, width = self.crop_size
        buffer = self.spatial_transform(
            images=buffer,
            target_height=height,
            target_width=width,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )



        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)


        buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)
        # img = buffer[:,0].numpy().transpose(1, 2, 0)
        # plt.imsave(f"images/img.png", img)
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
    C, T, H, W = tensor.shape
    tensor = tensor.contiguous().view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front

    
    return tensor

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, 3))
ISTD = np.array([1/0.229, 1/0.224, 1/0.225], dtype=np.float32).reshape((1, 1, 1, 3))

def randomize_video(frames, color_jitter=None, rand_gray=None, rand_crop=None, rand_rot=0, rand_trans=np.array([0,0]), normalize=False, rand_flip=False,force_flip=None):
    
    # frames = [fr.astype(np.float32) for fr in frames]
    # frames =  frames.astype(np.float32)
    
    
    
    if color_jitter is not None:
        # print('color_jitter')
        rand_h, rand_s, rand_v = [np.random.uniform(-h, h) for h in color_jitter]
        delta = np.array([rand_h * 180, rand_s, rand_v]).reshape((1, 1, 3)).astype(np.float32)
        frames = np.array([np.clip(cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_RGB2HSV) + delta, cv2.COLOR_HSV2RGB), 0, 255) for fr in frames])
        
    if rand_gray and np.random.uniform() < rand_gray:
        print('rand_gray')
        frames = [np.tile(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)[:,:,None], (1,1,3)) for fr in frames]

    applied_crop = [0]*4
    if rand_crop is not None:
        print('rand_crop')
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
        print('rand_trans')
        rot = np.random.uniform(-rand_rot, rand_rot)
        trans = np.random.uniform(-rand_trans, rand_trans)
        M = np.array([[np.cos(rot), -np.sin(rot), trans[0]], [np.sin(rot), np.cos(rot), trans[1]]])
        frames = [cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0])) for fr in frames]
    if rand_flip:
        print('rand_flip')
        if force_flip is not None:
            vert,horz = force_flip
        else:
            vert = -1 if random.random() > 0.5 else 1
            horz = -1 if random.random() > 0.5 else 1
        frames = [x[::vert,::horz] for x in frames]
        flip = np.array([[horz,0,0],[0,vert,0],[0,0,1]])
    if normalize:
        # print('normalize')
        frames = (frames/255 - MEAN)*ISTD
    else:
        frames = frames/255
    # frames = np.concatenate([fr[None] for fr in frames], 0).astype(np.float32)
    return frames, None


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