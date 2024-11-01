#!/usr/bin/env python

from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import QuickGELU, LayerNorm
from weight_loaders import weight_loader_fn_dict
from vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)

from collections import OrderedDict

class TempAttention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int, out_dim: int,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        self.return_all_features = return_all_features
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

        print('verbose...TempAttention init')


    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0); assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1); assert v.size(1) == Lkv # Lq=3136, Lkv=3136 (16*196)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k) #(B,1,3136,12)
        # q torch.Size([8, 3136, 12, 64]), k torch.Size([8, 3136, 12, 64]), v torch.Size([8, 3136, 12, 64])
       

        # ### add mask
        aff = aff.masked_fill(torch.logical_not(mask[:,:,:, None]), float('-inf'))
        # ###

        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        if self.return_all_features:
            return dict(q=q, k=k, v=v, aff=aff, out=out)
        else:
            return out



class TransformerEncoderLayerTempAtt(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.return_all_features = return_all_features

        self.attn = TempAttention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim,
            return_all_features=return_all_features,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)


    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, mask)
        x = x + self.mlp(self.norm2(x))

        return x


class EVLDecoderTempAtt(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
      
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerEncoderLayerTempAtt(
                in_feature_dim=in_feature_dim, qkv_dim=in_feature_dim, num_heads=num_heads, mlp_factor=mlp_factor,
                return_all_features=False,
            ) for _ in range(1)
        ])

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(1)]
        )

        self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(1)]
            )
       
        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


    def forward(self, frame_features: torch.Tensor, mask: torch.Tensor):
        frame_features = frame_features[:,:,1:,:] # we don't use cls token here

        N, T, L, C = frame_features.size()

        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)
        
        # temporal position encoding
        frame_features += self.temporal_pos_embed[0].view(1, T, 1, C)
            
        frame_features = frame_features.flatten(1, 2) # N, T * L, C

        frame_features = self.blocks[0](frame_features, mask)
        
        # a transformer block
        x = self.decoder_layers[0](x, frame_features)
        
        return x


class EVLTransformerTempAtt(nn.Module):

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
        cls_dropout: float = 0.5,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        backbone_config = self._create_backbone(backbone_name, backbone_type, backbone_path, backbone_mode)
        backbone_feature_dim = backbone_config['feature_dim']
        #backbone_spatial_size = tuple(x // y for x, y in zip(backbone_config['input_size'], backbone_config['patch_size']))

        self.decoder = EVLDecoderTempAtt(
            num_frames=num_frames,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            mlp_dropout=decoder_mlp_dropout,
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(backbone_feature_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(backbone_feature_dim, num_classes),
        )

        print('verbose...EVLTransformerTempAtt')


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
        backbone = self._get_backbone(x)

        B, C, T, H, W = x.size()

        ## TODO place holder
        N = 196*16
        mask = torch.randint(0,2,(B,N,N))
        mask = mask.to(x.device)
        ###

        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features[-1]['out'], mask)
        x = self.proj(x[:, 0, :])

        return x