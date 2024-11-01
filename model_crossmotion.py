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
        

import torch

def cross_patch_motion_v1_allneighbors(point_trajs_gt_coord, point_trajs_visibility_mask):
    """
    Computes cross-patch motion features for each spatio-temporal tokens.
    
    Args:
        point_trajs_gt_coord: Tensor of shape (B, M, T, 2) -> M trajectories point coordinates, normalized between [-1, 1]
        point_trajs_visibility_mask: Tensor of shape (B, M, T) -> 1 indicates point visible, 0 indicates point invisible

    Returns:
        cross_path_motion_feature: Tensor of shape (B, T, M, D), D = M*2 for all neighbors

    """
    B, M, T, _ = point_trajs_gt_coord.shape
    assert (point_trajs_visibility_mask.shape == (B, M, T))
    
    # Find indices where NaNs occur in point_trajs_gt_coord
    nan_indices = torch.isnan(point_trajs_gt_coord)
    
    # Set NaN values in point_trajs_gt_coord to 0
    point_trajs_gt_coord = torch.nan_to_num(point_trajs_gt_coord, nan=0.0)
    
    # Set corresponding entries in point_trajs_visibility_mask to 0
    point_trajs_visibility_mask[nan_indices[..., 0]] = 0  # Only need to check one coordinate (x or y) for NaN

    tensor_centers = point_trajs_gt_coord.unsqueeze(3)                                                       # (B, M, T, 2) -> (B, M, T, 1, 2)
    tensor_neighbors = point_trajs_gt_coord.permute(0, 2, 1, 3).unsqueeze(1)                 # (B, M, T, 2) -> (B, T, M, 2) -> (B, 1, T, M, 2)
    distances_relative = tensor_centers - tensor_neighbors                                    # (B, M, T, M, 2)

    # Create a visibility mask for each pair
    vis_mask_centers = point_trajs_visibility_mask.unsqueeze(3)                               # (B, M, T, 1)
    vis_mask_neighbors = point_trajs_visibility_mask.permute(0, 2, 1).unsqueeze(1)            # (B, 1, T, M)
    vis_mask_pair = vis_mask_centers * vis_mask_neighbors                                     # (B, M, T, M)

    # Apply the mask to the relative distances
    distances_relative = distances_relative * vis_mask_pair.unsqueeze(-1)                     # (B, M, T, M, 2)

    D = M * 2
    cross_path_motion_fea = distances_relative.reshape(B, M, T, D)                            # (B, M, T, D)
    cross_path_motion_fea = cross_path_motion_fea.permute(0, 2, 1, 3)

    return cross_path_motion_fea



class CrossmotionModule(nn.Module):

    def __init__(
        self,
        in_feature_dim = 512, # all neighbors 256x2
        fea_dim_absolute = 512,
        fea_dim_motion = 512,
        out_feature_dim = 768,
        num_patches = 196,
    ):
        super().__init__()
        self.num_patches = num_patches # we don't need cls token

        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, fea_dim_absolute]))

        self.fc1 = nn.Linear(in_feature_dim, fea_dim_motion)

        self.fc_out = nn.Linear(fea_dim_motion + fea_dim_absolute, out_feature_dim)
    
    def forward(self, point_trajs_gt_coord, point_trajs_visibility_mask):
        '''
        Args:
        point_trajs_gt_coord: Tensor of shape (B, M, T, 2) -> M trajectories point coordinates, normalized between [-1, 1]
        point_trajs_visibility_mask: Tensor of shape (B, M, T) -> 1 indicates point visible, 0 indicates point invisible

        Returns:
            cross_path_motion_feature: Tensor of shape (B, T, M, D), D = M*2 for all neighbors

        '''
        B, N, T, _ = point_trajs_gt_coord.shape

        # cross-patch motion cues
        motion_raw_feas = cross_patch_motion_v1_allneighbors(point_trajs_gt_coord, point_trajs_visibility_mask) # (B,T,N,D=512)

        motion_hidden_feas = self.fc1(motion_raw_feas) # (B, T, N, D2=256)

        # concate absolute position and cross-patch relative motion feas
        absolute_pos_embd = self.pos_embed.unsqueeze(0).unsqueeze(0)
        absolute_pos_embd = absolute_pos_embd.repeat(B, T, 1, 1)

        cat_feas = torch.concat((motion_hidden_feas, absolute_pos_embd),dim=-1) # (B, T, N, D3=256)

        motion_out_feas = self.fc_out(cat_feas) # (B, T, N, D4=256)

        return motion_out_feas
    


class EVLDecoderCrossmotion(nn.Module):

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
    def forward(self, frame_features, crosspatch_motion_features):
        frame_features = frame_features[:,:,1:,:] # we don't use cls token here

        B, T, N, C = frame_features.size()
        _, M, D = crosspatch_motion_features.shape
        assert (_ == B)

        x = self.cls_token.view(1, 1, -1).repeat(B, 1, 1)
        
        # temporal position encoding
        frame_features += self.temporal_pos_embed[0].view(1, T, 1, C)

        # add cross-patch motion encoding
        frame_features += crosspatch_motion_features
            
        frame_features = frame_features.flatten(1, 2) # B, T * N, C
        
        # a transformer block
        x = self.decoder_layers[0](x, frame_features)
        
        return x


class EVLTransformerCrossmotion(nn.Module):

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

        self.decoder = EVLDecoderCrossmotion(
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

        self.crossmotion_model = CrossmotionModule(in_feature_dim=14*14*2, fea_dim_absolute=512, fea_dim_motion=512)

        print('verbose...EVLTransformerCrossmotion')


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


    def forward(self, x):
        backbone = self._get_backbone(x)
        B, C, T, H, W = x.size()

        #TODO replace with real input
        metadata = {}
        metadata['base_points'] =  torch.randn(B, 196, T, 2).to(device=x.device)
        metadata['pt_visibility'] = torch.randint(0, 2, (B, 196, T)).to(device=x.device)

        point_trajs_gt_coord, point_trajs_visibility_mask = metadata['base_points'], metadata['pt_visibility']
        crosspatch_motion_features = self.crossmotion_model(point_trajs_gt_coord, point_trajs_visibility_mask )

       
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features[-1]['out'], crosspatch_motion_features)
        x = self.proj(x[:, 0, :])

        return x

if __name__ == "__main__":
    # Define a toy example input
    B, M, T = 2, 3, 5  # Batch size of 2, 3 points, 5 timesteps
    point_trajs_gt_coord = torch.randn(B, M, T, 2)  # Random tensor for coordinates
    point_trajs_visibility_mask = torch.randint(0, 2, (B, M, T))  # Random visibility mask with values 0 or 1

    # Run the function
    #output = cross_patch_motion_v1_allneighbors(point_trajs_gt_coord, point_trajs_visibility_mask)

    # Initialize the module
    cross_motion_module = CrossmotionModule(in_feature_dim=M*2,fea_dim_absolute=512,fea_dim_motion=256,out_feature_dim=768, num_patches=M)

    # Forward pass
    output = cross_motion_module(point_trajs_gt_coord, point_trajs_visibility_mask)
    print("Output shape:", output.shape)

    # Print the output shape
    print("Output shape:", output.shape)  # Expected shape: (B, T, M, D) where D = M * 2