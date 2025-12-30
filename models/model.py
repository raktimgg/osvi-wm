import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attentive_pooler import AttentivePooler
from models.basic_embeddings import ResNetFeats
from models.system_model import TransformerModel
from models.traj_embed import NonLocalLayer, TemporalPositionalEncoding
import copy
from einops import rearrange
from einops.layers.torch import Rearrange

def to_channel_first(images):
    return rearrange(images,'... r col c -> ... c r col')

def to_channel_last(images):
    return rearrange(images,'... c r col -> ... r col c')

class Encoder(nn.Module):
    def __init__(self, frozen=False, scratch=False):
        super().__init__()
        self.encoder = ResNetFeats(output_raw=True, drop_dim=2, use_resnet18=True,frozen=frozen,scratch=scratch)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad = False
        self.feat_dim = 512

    def forward(self, images, context, ret_resnet=False):
        assert len(images.shape) == 5, "expects [B, T, C, H, W] tensor!"
        enc_features_raw, target_features = self._encoder_features(images,context)

        enc_features = enc_features_raw
        target_features = target_features

        return enc_features, target_features, enc_features_raw

    def _encoder_features(self, images,context):
        im_in = torch.cat((context, images), 1)
        B, T, C, H, W = im_in.shape
        T_ctxt = context.shape[1]

        enc_im = im_in
        target_im = im_in[:,T_ctxt+1:]
        enc_features = self.encoder(enc_im).transpose(1, 2)
        with torch.no_grad():
            target_features = self.target_encoder(target_im).transpose(1, 2)

        enc_features = enc_features.transpose(1, 2)
        target_features = target_features.transpose(1, 2)
        return enc_features, target_features
    
    def _spatial_embed(self, features,ret_mask=False):

        features = F.softmax(features.reshape((features.shape[0], features.shape[1], features.shape[2], -1)), dim=3).reshape(features.shape)
        # could also get values (depth?) by doing torch.sum(features*softmaxed_features)
        h = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, features.shape[4]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 3), 3)
        if ret_mask:
            return torch.cat((h, w), 2), features
        else:
            return torch.cat((h, w), 2)

class ActionModel(nn.Module):
    def __init__(self, dropout, attn_heads, n_st_attn, trans_encoder=False):
        super().__init__()

        self.feat_dim = 512
        self._pe = TemporalPositionalEncoding(self.feat_dim, dropout)

        self._st_attn = nn.Sequential(NonLocalLayer(self.feat_dim, 512, 128, dropout=dropout, causal=True, n_heads=attn_heads, dino_feat=trans_encoder),
                                      *[NonLocalLayer(512, 512, 128, dropout=dropout, causal=True, n_heads=attn_heads) for _ in range(n_st_attn-2)],
                                      NonLocalLayer(512, self.feat_dim, 128, dropout=dropout, causal=True, n_heads=attn_heads))

    def forward(self, enc_features):
        enc_features2 = self._pe(enc_features.transpose(1, 2)).transpose(1, 2)
        act_features = self._st_attn(enc_features2.transpose(1, 2)).transpose(1, 2)
        return act_features

    def _spatial_embed(self, features,ret_mask=False):

        features = F.softmax(features.reshape((features.shape[0], features.shape[1], features.shape[2], -1)), dim=3).reshape(features.shape)
        # could also get values (depth?) by doing torch.sum(features*softmaxed_features)
        h = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, features.shape[4]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 3), 3)
        if ret_mask:
            return torch.cat((h, w), 2), features
        else:
            return torch.cat((h, w), 2)

class StateSpaceModel(nn.Module):
    def __init__(self, latent_dim, waypoints=None, sub_waypoints=False, metaworld=True):
        super().__init__()
        if sub_waypoints and waypoints is not None:
            waypoints = (waypoints+1)*( waypoints)//2
        self.waypoints = waypoints
    
        self._embed = Encoder(frozen=False, scratch=False)
        
        if metaworld:
            self.action_model = ActionModel(dropout=0.3, attn_heads=4, n_st_attn=6) # for metaworld
        else:
            self.action_model = ActionModel(dropout=0., attn_heads=4, n_st_attn=2)  # for pick place and franka
        
        in_dim = 512
        self._to_embed = nn.Sequential(nn.Linear(in_dim*2, latent_dim*5))
        self._to_waypoints = nn.Sequential(nn.Linear(in_dim, in_dim), 
                                           nn.ReLU(), 
                                           nn.Dropout(0.2),
                                           nn.Linear(in_dim, in_dim),)

        # determined based on feature map size
        if metaworld:
            factor = 49
        else:
            factor = 80
        
        self.forward_model = TransformerModel(input_dim=in_dim*factor + in_dim*factor,
                                            window_size=30,
                                            n_layer=1,
                                            n_head=4,
                                            n_embed=512,
                                            output_dim=in_dim*factor,
                                            dropout=0.3,
                                            # dropout=0.,
                                            bias=True,
                                            is_causal=True)

        self.attn_pe = TemporalPositionalEncoding(d_model=in_dim*2)
        self.attn_pool = AttentivePooler(num_queries=1, embed_dim=in_dim*2, num_heads=1, complete_block=False)

        self.waypoint_head = nn.Sequential(nn.Dropout(0.2),nn.ReLU(),nn.Linear(latent_dim*5,waypoints*4),Rearrange('... (w d) -> ... w d',d = 4))

    def forward(self, images, context, T_tot=None):
        T_context = context.shape[1]
        enc_features, target_features, enc_features_raw = self._embed(images, context, False)

        context_states = enc_features[:,:T_context]
        current_state = enc_features[:,T_context:T_context+1]
        all_next_states = []
        if T_tot == None:
            T_tot = enc_features.shape[1]
        for i in range(T_tot-T_context-1):
            act_input = torch.concat([context_states, current_state],dim=1)
            act_features = self.action_model(act_input)
            forward_model_input = torch.concat([act_features[:,T_context:], current_state],dim=2).flatten(2,4)
            next_state = self.forward_model(forward_model_input)[:,-1:]
            next_state = next_state.view(next_state.shape[0],1,enc_features.shape[2],enc_features.shape[3],enc_features.shape[4])
            current_state = torch.concat([current_state, next_state],dim=1)
            all_next_states.append(next_state)

        all_next_states = torch.cat(all_next_states, dim=1)
        forward_states = self._spatial_embed(all_next_states)
        waypoint_states = forward_states

        out = {}
        out['raw_enc_features'] = enc_features
        out['enc_features'] = self._spatial_embed(enc_features)
        out['raw_forward_states'] = all_next_states
        out['forward_states'] = forward_states
        out['target_features'] = target_features
        out['waypoint_states'] = waypoint_states[:,0]
        
        waypoint_states = self.attn_pe(waypoint_states)
        waypoint_states = self.attn_pool(waypoint_states)
        waypoint_states = waypoint_states.view(waypoint_states.shape[0],-1)
        act_embed = self._to_embed(waypoint_states)
        x = self.waypoint_head(act_embed)
        out['waypoints'] = x
        return out
    
    def _spatial_embed(self, features,ret_mask=False):

        features = F.softmax(features.reshape((features.shape[0], features.shape[1], features.shape[2], -1)), dim=3).reshape(features.shape)
        h = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, features.shape[4]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 3), 3)
        if ret_mask:
            return torch.cat((h, w), 2), features
        else:
            return torch.cat((h, w), 2)