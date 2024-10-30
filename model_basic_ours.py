#!/usr/bin/env python

from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#from vision_transformer import QuickGELU, Attention
from weight_loaders import weight_loader_fn_dict
from vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from einops import rearrange
from functools import partial


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class BasicTr(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 use_checkpoint=False,
                 use_mean_pooling=False,
                 ):
        super().__init__()
      
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.tubelet_size = tubelet_size
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        # num_patches = self.patch_embed.num_patches 
        self.use_checkpoint = use_checkpoint

        # if use_learnable_pos_emb:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # else:
        #     # sine-cosine positional embeddings is on the way
        #     self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x

    # original
    # def forward(self, x):
    #     _, T, _, _, _ = x.shape
    #     x = rearrange(x, 'b t p q d -> b (t p q) d')
    #     x = self.forward_features(x)
    #     x, x_patch_tokens = x[:, 0], x[:, 1:]
    #     return x

    def forward(self, x):
        _, T, _, _ = x.shape
        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.forward_features(x)
        x, x_patch_tokens = x[:, 0], x[:, 1:]
        x = self.head(self.fc_dropout(x))
        patch_tokens = rearrange(x_patch_tokens, 'b (t n) d -> b t n d', t=T)

        return x, patch_tokens
  

class OursTransformerBasic(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        backbone_name: str = 'ViT-B/16',
        backbone_type: str = 'clip',
        backbone_path: str = '',
        backbone_mode: str = 'frozen_fp16',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        num_classes: int = 400,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        cls_dropout: float = 0.5,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        backbone_config = self._create_backbone(backbone_name, backbone_type, backbone_path, backbone_mode)
        backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(x // y for x, y in zip(backbone_config['input_size'], backbone_config['patch_size']))


        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.decoder = BasicTr(
            num_classes=num_classes,
            depth=1,
            embed_dim = backbone_feature_dim, 
            num_heads=8,
            mlp_ratio=4, 
            qkv_bias=True, 
            norm_layer=norm_layer,
        )

        print('verbose...create basic module')

        # self.proj = nn.Sequential(
        #     nn.LayerNorm(backbone_feature_dim),
        #     nn.Dropout(cls_dropout),
        #     nn.Linear(backbone_feature_dim, num_classes),
        # )


    def _create_backbone(
        self,
        backbone_name: str,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer2D(return_all_features=True, **vit_presets[backbone_name])
        backbone.load_state_dict(state_dict, strict=True) # weight_loader_fn is expected to strip unused parameters

        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            self.backbone = [backbone] # avoid backbone parameter registration

        return vit_presets[backbone_name]


    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            # freeze backbone
            self.backbone[0] = self.backbone[0].to(x.device)
            return self.backbone[0]
        else:
            # finetune bakbone
            return self.backbone


    def forward(self, x: torch.Tensor):
        # x: ([B, 3, T, 224, 224])
        # 
        backbone = self._get_backbone(x)

        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

        ### 
        # features is a list of length 4, each item is a dict, keys dict_keys(['q', 'k', 'v', 'attn_out', 'out'])
        # each feature dimensions are the same
        # features[i]['q'].shape == k.shape ==v.shape== torch.Size([128, 197, 12, 64]) 
        # features[i]['attn_out'].shape == features[0]['out'].shape== torch.Size([128, 197, 768]) (B*T, N, D)

        features = backbone(x)[-self.decoder_num_layers:] 

        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]
        
        ###
        #  features[0]['q'].shape  torch.Size([8, 16, 197, 12, 64]) (B, T, N, heads, D0)
        #  features[0]['out'].shape  torch.Size([8, 16, 197, 768]) (B, T, N, D)

        x, _ = self.decoder(features[-1]['out']) # (B,1,D)

        return x
    





