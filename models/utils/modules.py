# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
import numpy as np
from einops import rearrange, repeat, parse_shape

from models.utils.tensors import drop_path, TruncatedNormal


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True,
        is_causal=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if self.use_sdpa:
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            print('Not using SDP Attention')
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        grid_size=None,
        grid_depth=None,
        is_causal=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x, return_attention=False):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)
        self.use_sdpa = False
        if self.use_sdpa:
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                q = F.scaled_dot_product_attention(q, k, v)
                xattn = None
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = (xattn @ v)


        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)
        # action_embedding = q[:, -1:, :]
    
        # if return_attention:
        return q, xattn
        # return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_sdpa=False)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x, return_attention=False):
        if return_attention:
            y, attn = self.xattn(q, self.norm1(x), return_attention)
        else:
            y = self.xattn(q, self.norm1(x), return_attention)
            attn = None
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q, attn
    

class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=True,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None, **kwargs):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist

    def loss_fn(self, dist, target, reduction="mean", **kwargs):
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

        return {
            "actor_loss": loss,
        }
    

class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=True, sep_var=False):
        super().__init__()
        assert n_mixtures  >= 1, "must predict at least one mixture!"
        self._n_mixtures   =  n_mixtures
        self._dist_size    =  torch.Size((out_dim, n_mixtures))
        self._mu           =  nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob   =  nn.Linear(in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None
        if const_var:
            ln_scale       = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if sep_var:
            ln_scale       = torch.randn((out_dim, n_mixtures), dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if not (const_var or sep_var):
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)

    def forward(self, x): # x has shape B T d
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))

        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            if len(ln_scale.shape) == 1:
                ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
                #(1, 1, 8, 1) -> (B T, dist_size[0], dist_size[1]) i.e. each mixture has the **same** constant variance
            else: # the sep_val case:
                ln_scale = repeat(ln_scale, 'out_d n_mix -> B T out_d n_mix', B=x.shape[0], T=x.shape[1])

        logit_prob = self._logit_prob(x).reshape(mu.shape) if self._n_mixtures > 1 else torch.ones_like(mu)
        return (mu, ln_scale, logit_prob)
    

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        #one_hot = one_hot.cuda()
        one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot    

class DiscreteMixLogistic(torch.distributions.Distribution):
    arg_constraints = {}
    def __init__(self, mean, log_scale, logit_probs, num_classes=256, log_scale_min=-7.0):
        assert mean.device == log_scale.device and mean.device == logit_probs.device, "all tensors must lie on same device!"
        batch_shape = log_scale.shape[:-1]
        event_shape = mean.shape[len(batch_shape)+1:]
        super().__init__(batch_shape, event_shape, None)
        self._mean = mean
        self._log_scale = log_scale
        self._logit_probs = logit_probs
        self._num_classes = num_classes
        self._log_scale_min = log_scale_min

    def log_prob(self, value):
        # reshape value to match convention
        B, n_mix = value.shape[0], self._log_scale.shape[-1]

        # unpack parameters. (B, T, num_mixtures) x 3
        logit_probs = self._logit_probs.reshape((self._log_scale.shape[0], -1, n_mix))
        means = self._mean.reshape((self._mean.shape[0], -1, n_mix))
        log_scales = torch.clamp(self._log_scale.reshape((self._log_scale.shape[0], -1, n_mix)), min=self._log_scale_min)

        # B x T x 1 -> B x T x num_mixtures
        y = value.reshape((B, -1, 1))

        centered_y = y - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_y + 1. / (self._num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1. / (self._num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0 (before scaling)
        # equivalent: torch.log(torch.sigmoid(plus_in))
        log_cdf_plus = plus_in - F.softplus(plus_in)

        # log probability for edge case of 255 (before scaling)
        # equivalent: (1 - torch.sigmoid(min_in)).log()
        log_one_minus_cdf_min = -F.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered_y
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # tf equivalent
        """
        log_probs = tf.where(x < -0.999, log_cdf_plus,
                            tf.where(x > 0.999, log_one_minus_cdf_min,
                                    tf.where(cdf_delta > 1e-5,
                                            tf.log(tf.maximum(cdf_delta, 1e-12)),
                                            log_pdf_mid - np.log(127.5))))
        """
        # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
        # for num_classes=65536 case? 1e-7? not sure..
        inner_inner_cond = (cdf_delta > 1e-5).float()

        inner_inner_out = inner_inner_cond * \
            torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
            (1. - inner_inner_cond) * (log_pdf_mid - np.log((self._num_classes - 1) / 2))
        inner_cond = (y > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = (y < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

        log_probs = log_probs + F.log_softmax(logit_probs, -1)
        return torch.logsumexp(log_probs, axis=-1).reshape(value.shape)


    def sample(self):
        n_mix = self._log_scale.shape[-1]
        logit_probs = self._logit_probs.reshape((self._log_scale.shape[0], -1, n_mix))

        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, n_mix)
        # select logistic parameters
        means = self._mean.reshape((self._mean.shape[0],-1, n_mix))
        means = torch.sum(means * one_hot, dim=-1)
        log_scales = self._log_scale.reshape((self._log_scale.shape[0], -1, n_mix))
        log_scales = torch.sum(log_scales * one_hot, dim=-1)

        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
        x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

        return x.reshape(self._mean.shape[:-1])
    
    @property
    def mean_(self):
        alphas = F.softmax(self._logit_probs, dim=-1)
        return torch.sum(self._mean * alphas, -1)