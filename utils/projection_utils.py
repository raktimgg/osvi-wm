import numpy as np
# updated to be [-1,1] on both axes (camera matrix has to be tweaked based on image ratio)
#
# normal image point is 0 in the center of the image, +x is right, +y is up
# both axes are [-1,1]
# normalized image point is x,y so need to flip axis and negate rows (because rows count down)
def image_point_to_pixels(imp, image_shape):
    if imp.shape[-1] > 2:
        imp = imp[...,:2]/imp[...,2:3]
    # fixed = (np.flip(imp[...,:2],axis=-1) + np.array(image_shape[:2])/image_shape[0]) * image_shape[0] / 2
    fixed = (np.flip(imp[...,:2],axis=-1) + 1) * image_shape[:2] / 2
    fixed[...,0] = image_shape[0] - fixed[...,0]
    return fixed.astype(int)

def pixels_to_image_point(point, image_shape):
    pc = point.copy()
    pc[0] = image_shape[0] - pc[0]
    # fixed = np.flip(pc[:2] * 2 / image_shape[0] - np.array(image_shape[:2])/image_shape[0],axis=-1)
    fixed = np.flip(pc[:2] * 2 / image_shape[:2] - 1,axis=-1)
    return fixed

def embed_matrix_square(mat,size=4):
    trans = np.eye(size)
    trans[:mat.shape[0],:mat.shape[1]] = mat
    return trans

def embed_mat(mat,size=4):
    if len(mat.shape) == 1:
        trans = np.ones((size,))
        trans[:mat.shape[0]] = mat
    else:
        trans = np.eye(size)
        trans[:mat.shape[0],:mat.shape[1]] = mat
    return trans


import torch
# project normal image coords (with predicted z values) into 3d coords
# requires the projection matrix be provided, this should be the matrix for homogenious image points to 3d
def image_coords_to_3d(image_coords,proj_mats):
    # convert normal image coords to homogenious coords by scaling by third component
    hom_im_coords = torch.cat((image_coords[...,:2]*image_coords[...,2:3],image_coords[...,2:3]),axis=-1)
    # add 1 to be homogenious for 4d conversion
    hom_im_4d = torch.cat((hom_im_coords, torch.ones(*hom_im_coords.shape[:-1],1).cuda()),axis=-1)
    # do the matrix multiplication
    trans_waypoints = torch.einsum('bwh,bdh->bwd',hom_im_4d,proj_mats)
    return trans_waypoints

def points_3d_to_image_hom_cords(points,matrix):
    points = np.concatenate((points,np.ones((*points.shape[:-1],1))),axis=-1)
    hom_pts = np.einsum('td,wd->tw',points,matrix)
    hom_pts = hom_pts[...,:3]/hom_pts[...,2:3]
    return hom_pts

to_01 = np.array([[0.5, 0., 0.5, 0.],
          [0., 0.5, 0.5, 0.],
          [0., 0., 1., 0.],
          [0., 0., 0., 1.]])
from_01 = np.linalg.inv(to_01)
# computes matrix which takes homogeneous 'normal image coords' [-1,1] on both axes
# and computes what the new image coords would be after cropping the image (crop specified)
# in pixels, so image size must be provided
def compute_crop_adjustment(crop_values,size_before):
    rt,rb,cl,cr = crop_values
    rows_before,cols_before = size_before
    row_scale = (rows_before-(rt+rb))/rows_before
    row_shift = rb/rows_before
    col_scale = (cols_before-(cl+cr))/cols_before
    col_shift = cl/cols_before
    crop_adjustment = np.array([[col_scale, 0., col_shift, 0.],
              [0., row_scale, row_shift, 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
    crop_mat = from_01 @ np.linalg.inv(crop_adjustment) @ to_01
    return crop_mat


def do_projection(pos,MVP_matrix,cam_pos,frame_width=224,frame_height=224):
    world_coord = np.ones((4, 1))
    world_coord[:3, 0] = pos - cam_pos
    image_coord = MVP_matrix.dot(world_coord)
    point = image_coord[:2].ravel()/image_coord[2]
    point = np.array([frame_height,frame_width])-point
    return np.flip(point).astype(int)

