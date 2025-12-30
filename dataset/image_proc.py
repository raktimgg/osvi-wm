import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
import torch 
import torchvision.transforms.functional as TVTransformsFunc 
from pyrr import Quaternion 


def xy_to_uv(xy, K):
    """
    Convert 3D points in camera coordinates to 2D points in image coordinates.
    Args:
        xy: Nx3 tensor of 3D points in camera coordinates
        K: 3x3 camera intrinsic matrix
    Returns:
        uv: Nx2 tensor of 2D points in image coordinates
    """
    # use numpy for matrix multiplication
    uv = np.dot(K, xy.T).T
    uv = uv[:, :2] / uv[:, 2:]
    return uv

def create_belief_map(
        image_resolution,
        # image size (width x height)
        coordsBelief,
        # list of points to draw in a Nx3 tensor
        sigma=2
        # the size of the point
        # returns a tensor of n_points x h x w with the belief maps
    ):

    # Input argument handling
    assert (
        len(image_resolution) == 2
    ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
        len(image_resolution)
    )
    # Camera intrinsic parameters
    f_x = 269.3  # Focal length in x
    f_y = 269.3  # Focal length in y
    c_x = 112.0  # Principal point x-coordinate
    c_y = 112.0  # Principal point y-coordinate
    K = np.array([[f_x, 0.0, c_x],
                [0.0, f_y, c_y],
                [0.0, 0.0, 1.0]])
    
    pointsBelief = xy_to_uv(coordsBelief, K)
    print(pointsBelief.shape)
    image_width, image_height = image_resolution
    image_transpose_resolution = (image_height, image_width)
    out = np.zeros((len(pointsBelief), image_height, image_width))

    w = int(sigma * 2)

    for i_point, point in enumerate(pointsBelief):
        pixel_u = int(point[0])
        pixel_v = int(point[1])
        array = np.zeros(image_transpose_resolution)

        # TODO makes this dynamics so that 0,0 would generate a belief map.
        if (
            pixel_u - w >= 0
            and pixel_u + w + 1 < image_width
            and pixel_v - w >= 0
            and pixel_v + w + 1 < image_height
        ):
            for i in range(pixel_u - w, pixel_u + w + 1):
                for j in range(pixel_v - w, pixel_v + w + 1):
                    array[j, i] = np.exp(
                        -(
                            ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                            / (2 * (sigma ** 2))
                        )
                    )
        out[i_point] = array

    return out

def get_keypoints_from_beliefmap(belief_maps):
    # Assuming belief_maps and gt_keypoints have shapes (B, 7, H, W)
    B, C, H, W = belief_maps.shape

    # Flatten the spatial dimensions (H, W) into a single dimension for argmax
    pred_flat = belief_maps.view(B, C, -1)  # (B, 7, H*W)

    # Get the argmax over the flattened dimension (which gives index in H*W)
    pred_max_idx = pred_flat.argmax(dim=-1)    # (B, 7)

    # Convert the 1D indices into 2D coordinates (row, col)
    pred_y = pred_max_idx // W                 # (B, 7), row coordinate
    pred_x = pred_max_idx % W                  # (B, 7), column coordinate

    # Stack the coordinates to get shape (B, 7, 2) for both predicted and ground truth keypoints
    pred_coords = torch.stack((pred_x, pred_y), dim=-1).float()  # (B, 7, 2)
    return pred_coords

# Function to filter keypoints based on whether they are inside the image bounds (camera's frustum)
def filter_keypoints_in_frustum(gt_keypoints, pred_keypoints, image_width=640, image_height=480):
    valid_indices = np.where(
        (gt_keypoints[:, 0] >= 0) & (gt_keypoints[:, 0] <= image_width) &
        (gt_keypoints[:, 1] >= 0) & (gt_keypoints[:, 1] <= image_height)
    )[0]
    return gt_keypoints[valid_indices], pred_keypoints[valid_indices]