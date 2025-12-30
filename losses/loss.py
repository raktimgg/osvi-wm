import random
import numpy as np
import torch
import einops
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from losses.soft_dtw_cuda import SoftDTW


def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def off_diag_cov_loss(x: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(einops.rearrange(x, "... E -> E (...)"))
    return off_diag(cov).square().mean()


def simclr_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Computes the Supervised Contrastive Loss (SCL).
    
    Args:
        embeddings (torch.Tensor): Normalized embeddings of shape (B, D).
        labels (torch.Tensor): Class labels of shape (B,).
        temperature (float): Temperature scaling factor.
        
    Returns:
        torch.Tensor: Supervised contrastive loss.
    """
    embeddings = rearrange(embeddings, 'b t f -> (b t) f')
    labels = rearrange(labels, 'b t -> (b t)')
    
    B = embeddings.shape[0]  # Batch size
    
    # Normalize embeddings to unit norm (important for cosine similarity)
    embeddings = F.normalize(embeddings, dim=-1)

    # Compute cosine similarity between all pairs (B x B)
    sim_matrix = torch.mm(embeddings, embeddings.T) / temperature

    # Create mask where mask[i, j] = 1 if labels[i] == labels[j], else 0
    labels = labels.view(B, 1)  # Ensure shape (B, 1)
    mask = (labels == labels.T).float()  # (B x B) mask of positive pairs

    # Remove self-similarity (diagonal elements should not contribute)
    mask.fill_diagonal_(0)

    # Compute log-sum-exp for denominator (all similarities)
    exp_sim = torch.exp(sim_matrix)  # e^(cosine similarity)
    denom = exp_sim.sum(dim=1, keepdim=True)  # Sum over all elements in a row

    # Compute numerator: sum of exponentials of positive pairs
    num = (exp_sim * mask).sum(dim=1, keepdim=True)

    # Compute loss using log sum exp trick
    loss = -torch.log(num / denom + 1e-8).mean()

    return loss

def compute_infonce_loss(embeddings, labels):
    """
    Compute InfoNCE loss using pytorch-metric-learning with custom sampling.

    Args:
        latent_states_aug (torch.Tensor): Shape (B, T, F)
        labels (torch.Tensor): (B, T).
    Returns:
        torch.Tensor: InfoNCE loss.
    """
    embeddings = rearrange(embeddings, 'b t f -> (b t) f')
    labels = rearrange(labels, 'b t -> (b t)')
    # print(embeddings.shape, labels.shape)
    loss_fn = losses.NTXentLoss(temperature=0.5)
    loss = loss_fn(embeddings, labels=labels)

    return loss


def sdtw_trajectory_loss(waypoints,traj_points,projection,num_points,gamma,waypoint_nun,image_waypoints=False):
    cureepos = traj_points[:,0]
    future_poses = traj_points
    all_waypoints = waypoints
    device = waypoints.device
    point_sets = []
    transind = 0
    if image_waypoints:
        # convert normal image coords to homogenious coords by scaling by third component
        hom_im_coords = torch.cat((all_waypoints[:,:,:2]*all_waypoints[:,:,2:3],all_waypoints[:,:,2:3]),axis=-1)
        # add 1 to be homogenious for 4d conversion
        hom_im_4d = torch.cat((hom_im_coords, torch.ones(*hom_im_coords.shape[:2],1,device=device)),axis=-1)
        # do the matrix multiplication
        trans_waypoints = torch.einsum('bwh,bdh->bwd',hom_im_4d,projection.float())  # projection is T_camera_in_world
        # replace the first 3d while keeping the higher dimensions
        all_waypoints = torch.cat((trans_waypoints[:,:,:3],all_waypoints[:,:,3:]),axis=-1)
    waypoint_counts = list(range(1,waypoint_nun+1))
    for num_waypoints in waypoint_counts:
        prev_point = cureepos
        inter_points = []
        for i in range(num_waypoints):
            waypoints = all_waypoints[:,transind]
            diffs = waypoints-prev_point
            # only include endpoint for last waypoint
            points = torch.tensor(np.linspace(0,1,num=num_points//num_waypoints,endpoint=(i==(num_waypoints-1))),device=device)
            inter_points_raw = torch.einsum('bd,t->btd',diffs,points)+prev_point.unsqueeze(1)
            # set grasp value to fixed to waypoint value, so the whole segment has same value
            inter_points_raw[...,-1] = waypoints[...,-1].unsqueeze(-1)
            inter_points.append(inter_points_raw)
            prev_point=waypoints
            transind += 1
        inter_points = torch.cat(inter_points,dim=1)
        point_sets.append(inter_points)
    point_sets = torch.stack(point_sets)

    future_poses_batched = repeat(future_poses,'b t d -> (w b) t d',w=point_sets.shape[0])
    point_sets_batched = rearrange(point_sets,'w b t d -> (w b) t d')
    # higher gamma values results in overshooting if there is a cluster of states at the end
    # (i.e. the gripper sitting in the goal pos). The right thing might be to anneal this value
    # to 0 over training
    sdtw = SoftDTW(use_cuda=True, gamma=gamma)
    loss = sdtw(future_poses_batched,point_sets_batched)
    # logging
    log_items = {}
    log_items['loss_batch'] = rearrange(loss,'(w b) -> b w',w=point_sets.shape[0]).mean(axis=1)
    log_items['gamma'] = gamma
    losses = rearrange(loss,'(w b) -> w b', w=point_sets.shape[0]).mean(axis=1)
    for i,counts in enumerate(waypoint_counts):
        log_items[f'loss_w{counts-1}'] = losses[i].item()
    return loss.mean(axis=0), log_items


def sdtw_feature_loss(waypoints,traj_points,gamma):
    sdtw = SoftDTW(use_cuda=True, gamma=gamma)
    loss = sdtw(traj_points,waypoints)
    # Log some basic items.
    log_items = {
        'loss': loss.mean().item(),
        'gamma': gamma,
    }
    return loss.mean(axis=0), log_items