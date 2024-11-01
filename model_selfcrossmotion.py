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
    model_to_fp16, vit_presets, TransformerEncoderLayer, QuickGELU
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


def delta_trajectory_motion_v1(point_trajs_gt_coord, point_trajs_visibility_mask):
    '''
        Args:
        point_trajs_gt_coord: Tensor of shape (B, M, T, 2) -> M trajectories point coordinates, normalized between [-1, 1]
        point_trajs_visibility_mask: Tensor of shape (B, M, T) -> 1 indicates point visible, 0 indicates point invisible
        Returns:
            cross_path_motion_feature: Tensor of shape (B, M, T, 2),

    '''
    B, M, T, _ = point_trajs_gt_coord.shape
    assert (point_trajs_visibility_mask.shape == (B, M, T))

    # motion with a trajectory
    # Compute deltas as before
    deltaT = 1
    point_trajs_delta_coords = point_trajs_gt_coord[:, :, deltaT:, :] - point_trajs_gt_coord[:, :, :-1, :]

    # Create a visibility mask for deltas
    # Both the current and the previous points must be visible; otherwise, set delta to 0
    visibility_current = point_trajs_visibility_mask[:, :, deltaT:]
    visibility_previous = point_trajs_visibility_mask[:, :, :-1]
    visible_both = visibility_current * visibility_previous

    # Extend visibility mask to match delta coordinates shape
    visible_both_extended = visible_both.unsqueeze(-1).expand_as(point_trajs_delta_coords)

    # Apply the visibility mask
    point_trajs_delta_coords *= visible_both_extended

    # Create initial zero movement with the correct shape and device
    initial_movement = torch.zeros(B, M, 1, 2, device=point_trajs_gt_coord.device)

    # Concatenate the zero initial movement with the masked deltas
    point_trajs_delta_coords_full = torch.cat([initial_movement, point_trajs_delta_coords], dim=2)

    assert (point_trajs_delta_coords_full.shape == (B,M,T,2))
    
    return point_trajs_delta_coords_full

class SelfCrossMotionModule(nn.Module):

    def __init__(
        self,
        in_feature_dim = 768, # all neighbors 256x2
        fea_dim_absolute = 288,
        fea_dim_crossmotion = 256,
        fea_dim_selfmotion =256,
        out_feature_dim = 768,
        num_patches = 196,
        num_heads = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.5,
        num_layers=2,
        act: nn.Module = QuickGELU,
    ):
        super().__init__()
        self.num_patches = num_patches # we don't need cls token

        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, fea_dim_absolute]))

        self.fc1 = nn.Linear(in_feature_dim, fea_dim_crossmotion)

        self.fc2 = nn.Linear(2, fea_dim_selfmotion) #x,y delta coordinates as input

        #self.fc_out = nn.Linear(fea_dim_selfmotion + fea_dim_crossmotion + fea_dim_absolute, out_feature_dim)
        assert (fea_dim_absolute+fea_dim_crossmotion+fea_dim_selfmotion == out_feature_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                in_feature_dim=out_feature_dim, qkv_dim=out_feature_dim, num_heads=num_heads, mlp_factor=mlp_factor, act=act,
                return_all_features=False,
            ) for _ in range(num_layers)
        ])

    
    def forward(self, point_trajs_gt_coord, point_trajs_visibility_mask, point_trajs_gt_rgbtoken_index, temporal_pos_embed):
        '''
        Args:
        point_trajs_gt_coord: Tensor of shape (B, M, T, 2) -> M trajectories point coordinates, normalized between [-1, 1]
        point_trajs_visibility_mask: Tensor of shape (B, M, T) -> 1 indicates point visible, 0 indicates point invisible
        point_trajs_gt_rgbtoken_index: Tensor of shape (B, M, T) -> M trajectories indices, each index corresponds to RGB tokens, range [0, 195]
        temporal_pos_embed: tensor of shape (T, C)
        Returns:
            cross_path_motion_feature: Tensor of shape (B, T, M, D), D = M*2 for all neighbors

        '''
        B, M, T, _ = point_trajs_gt_coord.shape
        assert (point_trajs_gt_rgbtoken_index.shape == (B, M, T))
        assert (point_trajs_visibility_mask.shape == (B, M, T))
        assert (temporal_pos_embed.shape[0] == T)
        T, C = temporal_pos_embed.shape

        #### range assertion
        # Check if all values are between 0 and 1, inclusive
        assert point_trajs_visibility_mask.max() <= 1.0 and point_trajs_visibility_mask.min() >= 0.0, "Tensor contains values outside the range [0, 1]."
        
        # Check if all values are within the range [0, self.num_patches-1]
        assert point_trajs_gt_rgbtoken_index.min() >= 0 and point_trajs_gt_rgbtoken_index.max() <= self.num_patches - 1, "Index out of allowed range."
        
        ######### handle NAN here
        # Find indices where NaNs occur in point_trajs_gt_coord
        nan_indices = torch.isnan(point_trajs_gt_coord)
        
        # Set NaN values in point_trajs_gt_coord to 0
        point_trajs_gt_coord = torch.nan_to_num(point_trajs_gt_coord, nan=0.0)
        
        # Set corresponding entries in point_trajs_visibility_mask to 0
        point_trajs_visibility_mask[nan_indices[..., 0]] = 0  # Only need to check one coordinate (x or y) for NaN


        ######### cross-patch motion cues
        motion_raw_feas = cross_patch_motion_v1_allneighbors(point_trajs_gt_coord, point_trajs_visibility_mask) # (B,T,M,D=196*2)
        motion_hidden_feas = self.fc1(motion_raw_feas) # (B, T, M, D2=256)
     
        ########### obtain absolute position encoding
        N, D_abs = self.pos_embed.shape
        indexing = point_trajs_gt_rgbtoken_index.permute(0,2,1).reshape(B*T, M)
        # No need to repeat, use gather to directly fetch necessary entries. output: (B*T, M, D_abs)
        absolute_pos_embd_final = torch.gather(self.pos_embed.expand(B*T, -1, -1), 1, indexing.unsqueeze(-1).expand(-1, -1, D_abs))
        absolute_pos_embd_out = absolute_pos_embd_final.view(B, T, M, D_abs)

        
        ########## self motion delta within a trajectory
        point_trajs_delta_coords_full = delta_trajectory_motion_v1(point_trajs_gt_coord, point_trajs_visibility_mask) # (B, M, T, 2)
        selfmotion_feas = self.fc2(point_trajs_delta_coords_full) #  # (B, M, T, D4)
        selfmotion_feas = selfmotion_feas.permute(0, 2, 1, 3)  # (B, T, M, D4)

        # concate absolute position and cross-patch relative motion feas #TODO scale issue
        cat_feas = torch.concat((motion_hidden_feas, selfmotion_feas, absolute_pos_embd_out),dim=-1) # (B, T, M, D3=256)
        cat_feas += temporal_pos_embed.view(1, T, 1, C) #(B, T, M, C)

        ########### spatial self attention within a frame
        cat_feas_spatial = cat_feas.reshape(B*T, M, C)
        motion_out_feas_spatial = self.blocks[0](cat_feas_spatial) # (B*T, M, D4=256)
        
        ########### temporal attention across frame
        motion_in_feas_temporal = motion_out_feas_spatial.reshape(B, T, M, C).permute(0,2,1,3).reshape(B*M, T, C)
        motion_out_feas_temporal = self.blocks[1](motion_in_feas_temporal)
        
        ########### final output
        motion_out_feas = motion_out_feas_temporal.reshape(B,M,T,C).permute(0,2,1,3)
        assert (motion_out_feas.shape==(B, T, M, C))

        # if torch.isnan(motion_out_feas).any():
        #     breakpoint()  # Trig

        return motion_out_feas    



class EVLDecoderSelfCrossmotion(nn.Module):

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
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(2)]
        )

        self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(1)]
            )
       
        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))

        self.motion_model = SelfCrossMotionModule(in_feature_dim=14*14*2, fea_dim_absolute = 288, fea_dim_crossmotion = 288, fea_dim_selfmotion=192)


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


    #def forward(self, in_features: List[Dict[str, torch.Tensor]]):
    def forward(self, frame_features, pt_coords, pt_visibility, pt_grid_indices):
        frame_features = frame_features[:,:,1:,:] # we don't use cls token here

        B, T, N, C = frame_features.size()

        x = self.cls_token.view(1, 1, -1).repeat(B, 1, 1)
        
        # temporal position encoding
        frame_features += self.temporal_pos_embed[0].view(1, T, 1, C)
        frame_features = frame_features.flatten(1, 2) # B, T * N, C
        
        # a transformer block
        x = self.decoder_layers[0](x, frame_features)

        # motion features
        motion_features = self.motion_model(pt_coords, pt_visibility, pt_grid_indices, self.temporal_pos_embed[0]) #(B, T, M, C)
        motion_features = motion_features.flatten(1, 2) # B, T * M, C

        # a transformer block
        x = self.decoder_layers[1](x, motion_features)

        # # Check for NaN values
        # if torch.isnan(x).any():
        #     breakpoint()  # Trig
        #     # Find indices of NaN values
        #     nan_indices = torch.where(torch.isnan(x))

        return x


class EVLTransformerSelfCrossmotion(nn.Module):

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

        self.decoder = EVLDecoderSelfCrossmotion(
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

        
        print('verbose...EVLTransformerSelfCrossmotion')


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


    #def forward(self, x, pt_coords, pt_grid_indices, pt_visibility):

    def forward(self, x):
        backbone = self._get_backbone(x)
        B, C, T, H, W = x.size()

        #TODO place holder
        M = 196
        N = 196
        pt_coords = torch.rand(B, M, T, 2) * 2 - 1  # Normalized between [-1, 1]
        pt_visibility = torch.randint(0, 2, (B, M, T))
        pt_grid_indices = torch.randint(0, N, (B, M, T))  # Assuming RGB tokens range is within trajectory indices
   

        pt_coords = pt_coords.to(x.device)
        pt_visibility = pt_visibility.to(x.device)
        pt_grid_indices = pt_grid_indices.to(x.device)  # Assuming RGB tokens range is within trajectory indices
   
       
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features[-1]['out'], pt_coords,  pt_visibility, pt_grid_indices)
        x = self.proj(x[:, 0, :])

        return x

if __name__ == "__main__":
    # Parameters for mock data
    B, M, T, C = 3, 200, 16, 768  # Example dimensions
    N = 196

    # Generate mock data
    point_trajs_gt_coord = torch.rand(B, M, T, 2) * 2 - 1  # Normalized between [-1, 1]
    point_trajs_visibility_mask = torch.randint(0, 2, (B, M, T))
    point_trajs_gt_rgbtoken_index = torch.randint(0, N, (B, M, T))  # Assuming RGB tokens range is within trajectory indices
    temporal_pos_embed = torch.rand(T, C)

    # Create model instance
    model = SelfCrossMotionModule(in_feature_dim=M*2, fea_dim_absolute = 288, fea_dim_crossmotion = 288, fea_dim_selfmotion=192)

    # Test the model
    output = model(point_trajs_gt_coord, point_trajs_visibility_mask, point_trajs_gt_rgbtoken_index, temporal_pos_embed)
    print("Output shape:", output.shape)