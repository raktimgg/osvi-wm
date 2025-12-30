import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicEmbeddingModel(nn.Module):
    def __init__(self, k=3, batch_norm=True):
        super(BasicEmbeddingModel, self).__init__()
        norm_factory = lambda f: nn.InstanceNorm2d(f, affine=True) if batch_norm else lambda x: x

        self._conv_1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self._conv_1_n = norm_factory(64)

        self._conv_2_1 = nn.Conv2d(64, 128, k, padding=1)
        self._conv_2_1_n = norm_factory(128)
        self._conv_2_2 = nn.Conv2d(128, 128, k, padding=1)
        self._conv_2_2_n = norm_factory(128)
        self._pool_2 = nn.MaxPool2d(2)

        self._conv_3_1 = nn.Conv2d(128, 256, k, padding=1)
        self._conv_3_1_n = norm_factory(256)
        self._conv_3_2 = nn.Conv2d(256, 256, k, padding=1)
        self._conv_3_2_n = norm_factory(256)
        self._pool_3 = nn.MaxPool2d(2)

        self._conv_4_1 = nn.Conv2d(256, 512, k, padding=1)
        self._conv_4_1_n = norm_factory(512)
        self._conv_4_2 = nn.Conv2d(512, 512, k, padding=1)
        self._conv_4_2_n = norm_factory(512)
        self._pool_4 = nn.MaxPool2d(2)

        self._conv_5_1 = nn.Conv3d(512, 512, k, padding=(2, 1, 1))
        self._conv_5_1_n = nn.InstanceNorm3d(512, affine=True) if batch_norm else lambda x: x
        self._conv_5_2 = nn.Conv3d(512, 512, k, padding=(2, 1, 1))
        self._conv_5_2_n = nn.InstanceNorm3d(512, affine=True) if batch_norm else lambda x: x
        assert k % 2 == 1, "context must be odd"
        self._k = k 

        self._fc6_1 = nn.Linear(512, 512)
        self._fc6_2 = nn.Linear(512, 512)
        self._embed = nn.Linear(512, 128)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape((B * T, C, H, W))
        
        x = F.relu(self._conv_1_n(self._conv_1(x)))

        x = F.relu(self._conv_2_1_n(self._conv_2_1(x)))
        x = F.relu(self._conv_2_2_n(self._conv_2_2(x)))
        x = self._pool_2(x)

        x = F.relu(self._conv_3_1_n(self._conv_3_1(x)))
        x = F.relu(self._conv_3_2_n(self._conv_3_2(x)))
        x = self._pool_3(x)

        x = F.relu(self._conv_4_1_n(self._conv_4_1(x)))
        x = F.relu(self._conv_4_2_n(self._conv_4_2(x)))
        x = self._pool_4(x)

        x = torch.transpose(x.reshape((B, T, 512, x.shape[-2], x.shape[-1])), 1, 2)
        x = F.relu(self._conv_5_1_n(self._conv_5_1(x)))[:,:,:-2,:,:]
        x = F.relu(self._conv_5_2_n(self._conv_5_2(x)))[:,:,:-2,:,:]
        x = [x[:, :, max(0, t - self._k // 2): min(T, t + self._k // 2 + 1),:,:] for t in range(T)]
        x = [torch.max(torch.max(torch.max(t, 2)[0], 2)[0], 2)[0][:, None] for t in x]
        x = torch.cat(x, 1)
        
        x = x.reshape((B * T, 512))
        x = F.relu(self._fc6_1(x))
        x = F.relu(self._fc6_2(x))

        embed = self._embed(x)
        embed = embed.reshape((B, T, 128))

        return embed


class ResNetFeats(nn.Module):
    def __init__(self, out_dim=256, output_raw=False, drop_dim=1, use_resnet18=False,frozen=False,scratch=False):
        super(ResNetFeats, self).__init__()
        if not scratch:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if use_resnet18 else models.resnet50(pretrained=(not scratch))
        else:
            resnet = models.resnet18(weights=None) if use_resnet18 else models.resnet50(weights=models.ResNet50_Weights.SCRATCH)
        self._features = nn.Sequential(*list(resnet.children())[:-drop_dim])
        self._output_raw = output_raw
        self._out_dim = 512 if use_resnet18 else 2048
        self._out_dim = int(self._out_dim / 2 ** (drop_dim - 2)) if drop_dim >= 2 else self._out_dim
        if not output_raw:
            self._nn_out = nn.Sequential(nn.Conv2d(self._out_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), nn.Conv2d(out_dim, out_dim, 1))
            self._out_dim = out_dim
        if frozen:
            for param in self._features.parameters():
                param.requires_grad = False
    
    def forward(self, inputs, depth=None):
        reshaped = len(inputs.shape) == 5
        x = inputs.reshape((-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]))

        out = self._features(x)
        out = self._nn_out(out) if not self._output_raw else out 
        NH, NW = out.shape[-2:]

        # reshape to proper dimension
        out = out.reshape((inputs.shape[0], inputs.shape[1], self._out_dim, NH, NW)) if reshaped else out
        if NH * NW == 1:
            out = out[:,:,:,0,0] if reshaped else out[:,:,0,0]
        return out

    @property
    def dim(self):
        return self._out_dim


class CoordConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding=1, stride=1):
        super().__init__()
        self._conv = nn.Conv2d(in_dim + 2, out_dim, kernel_size, stride, padding)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h_pad = torch.linspace(-1, 1, H).reshape((1, 1, H, 1)).repeat((B, 1, 1, W))
        w_pad = torch.linspace(-1, 1, W).reshape((1, 1, 1, W)).repeat((B, 1, H, 1))
        x = torch.cat((x, h_pad.to(x.device), w_pad.to(x.device)), 1)
        return self._conv(x)


class VGGFeats(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self._vgg = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:9])
        
        def _conv_block(in_dim, out_dim, N):
            # pool and coord conv
            cc = CoordConv(in_dim, out_dim, 2, stride=2, padding=0)
            n, a = nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
            ops = [cc, n, a]

            for _ in range(N):
                c = nn.Conv2d(out_dim, out_dim, 3, padding=1)
                n, a = nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
                ops.extend([c, n, a])
            return ops
        
        self._v1 = nn.Sequential(*_conv_block(128, 256, 3))
        self._v2 = nn.Sequential(*_conv_block(256, 512, 3))
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._out_dim = out_dim
        self._linear = nn.Sequential(nn.Linear(512, out_dim), nn.ReLU(inplace=True))

    def forward(self, x, depth=None):
        reshaped = len(x.shape) == 5
        in_x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])) if reshaped else x
        vis_feat = self._pool(self._v2(self._v1(self._vgg(in_x))))
        out_feat = self._linear(vis_feat[:,:,0,0])
        out_feat = out_feat.view((x.shape[0], x.shape[1], self._out_dim)) if reshaped else out_feat
        return out_feat

    @property
    def dim(self):
        return self._out_dim


class _BottleneckConv(nn.Module):
    def __init__(self, in_dim, out_dim, feed_forward, in_k=1, in_stride=1, in_pad=0):
        super().__init__()
        self._c1 = nn.Conv2d(in_dim, feed_forward, in_k, stride=in_stride, padding=in_pad)
        self._a1 = nn.ReLU(inplace=True)
        self._c2 = nn.Conv2d(feed_forward, feed_forward, 3, stride=1, padding=1)
        self._a2 = nn.ReLU(inplace=True)
        self._c3 = nn.Conv2d(feed_forward, out_dim, 1, stride=1, padding=0)
        self._n = nn.InstanceNorm2d(out_dim)
        self._residual = in_dim == out_dim

    def forward(self, x):
        inter = self._c3(self._a2(self._c2(self._a1(self._c1(x)))))
        if self._residual:
            return self._n(x + inter)
        return self._n(x)


class SimpleSpatialSoftmax(nn.Module):
    def __init__(self, added_convs=3, out_dim=64):
        super().__init__()
        n_keep = 16 if added_convs else 15
        vgg_feats =list( models.vgg16(pretrained=True).features[:n_keep].children())
        added_feats = [_BottleneckConv(256, 256, 64) for _ in range(added_convs)]
        self._conv_stack = nn.Sequential(*(vgg_feats + added_feats))
        self._out = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, out_dim))

    def forward(self, x):
        reshaped = len(x.shape) == 5
        if reshaped:
            B, T, C, H, W = x.shape
            x = x.view((B * T, C, H, W))
        else:
            B, C, H, W = x.shape
            T = 1
        
        x = self._conv_stack(x)
        x = F.softmax(x.view((B*T, x.shape[1], -1)), dim=2).view((B*T, x.shape[1], x.shape[2], x.shape[3]))
        h = torch.sum(torch.linspace(-1, 1, x.shape[2]).view((1, 1, -1)).to(x.device) * torch.sum(x, 3), 2)
        w = torch.sum(torch.linspace(-1, 1, x.shape[3]).view((1, 1, -1)).to(x.device) * torch.sum(x, 2), 2)
        x = torch.cat((h, w), 1)
        x = x.reshape((B, T, 512)) if reshaped else x
        return self._out(x)


class BasicCNN(nn.Module):
    def __init__(self, drop_dim=None):
        super().__init__()
        c1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        a1 = nn.ReLU(inplace=True)
        c2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        a2 = nn.ReLU(inplace=True)
        c3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        a3 = nn.ReLU(inplace=True)
        c4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self._v_net = nn.Sequential(c1, a1, c2, a2, c3, a3, c4)
    
    def forward(self, inputs):
        reshaped = len(inputs.shape) == 5
        x = inputs.reshape((-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1])) if reshaped else inputs
        out = self._v_net(x)
        C, H, W = out.shape[-3:]
        out = out.reshape((inputs.shape[0], inputs.shape[1], C, H, W)) if reshaped else out
        return out

import torchvision.transforms as T
import numpy as np
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
class R3M(nn.Module):
    # scratch, and output raw do nothing
    def __init__(self, output_raw=True, drop_dim=1, use_resnet18=False,frozen=False,scratch=False):
        super(R3M, self).__init__()
        import r3m
        resnet = r3m.load_r3m('resnet18' if use_resnet18 else 'resnet50')
        resnet = resnet.module.convnet
        # self.invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ], std = 1/STD),
                               # T.Normalize(mean = -MEAN, std = [ 1., 1., 1. ])])
        self._features = nn.Sequential(*list(resnet.children())[:-drop_dim])
        self._out_dim = 512
        if frozen:
            for param in self._features.parameters():
                param.requires_grad = False
    
    def forward(self, inputs):
        reshaped = len(inputs.shape) == 5
        x = inputs.reshape((-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]))

        # inverse transform because r3m wants [0-255]
        # x = self.invTrans(x)
        # out = self._features(x*255)
        out = self._features(x)
        NH, NW = out.shape[-2:]

        # reshape to proper dimension
        out = out.reshape((inputs.shape[0], inputs.shape[1], self._out_dim, NH, NW)) if reshaped else out
        if NH * NW == 1:
            out = out[:,:,:,0,0] if reshaped else out[:,:,0,0]
        return out

class TemporalPositionalEncoding(nn.Module):
    """
    Modified PositionalEncoding from Pytorch Seq2Seq Documentation
    source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert len(x.shape) >= 3, "x requires at least 3 dims! (B, C, ..)"
        old_shape = x.shape
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x + self.pe[:,:,:x.shape[-1]]
        return self.dropout(x).reshape(old_shape)