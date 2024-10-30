#!/usr/bin/env python

from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import QuickGELU, Attention
from weight_loaders import weight_loader_fn_dict
from vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)
        


class EVLDecoderBasic(nn.Module):

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


        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(1)]
        )

        self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(1)]
            )
       
        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


    #def forward(self, in_features: List[Dict[str, torch.Tensor]]):
    def forward(self, frame_features: torch.Tensor):
        N, T, L, C = frame_features.size()

        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)
        
        # temporal position encoding
        frame_features += self.temporal_pos_embed[0].view(1, T, 1, C)
            
        frame_features = frame_features.flatten(1, 2) # N, T * L, C
        
        # a transformer block
        x = self.decoder_layers[0](x, frame_features)
        
        return x


class EVLTransformerBasic(nn.Module):

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

        self.decoder = EVLDecoderBasic(
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

        print('verbose...EVLTransformerBasic')


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
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features[-1]['out'])
        x = self.proj(x[:, 0, :])

        return x