import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from itertools import combinations, product

def get_valid_points(pred_tracks, per_point_queries, pred_visibility, cropped_coords):
    min_x, max_x, min_y, max_y = cropped_coords
    all_point_indices = np.arange(pred_tracks.shape[1]).astype(np.int16)
    init_point_each_point = pred_tracks[per_point_queries, all_point_indices]
    valid_x_points =  (init_point_each_point[:, 0] > min_x) & (init_point_each_point[:, 0] < max_x)
    valid_y_points = (init_point_each_point[:, 1] > min_y) & (init_point_each_point[:, 1] < max_y)
    valid_points = valid_x_points & valid_y_points
    valid_points = valid_points.numpy()
    return valid_points


def get_temporal_connection_data(pt_dict, num_patches, return_mask=False):
    # assert points.shape[0] == cfg.DATA.NUM_FRAMES
    points = pt_dict['pred_tracks']
    pt_visibility_mask = pt_dict['pred_visibility']
    temporal_length = points.shape[0] # not using frames due to downsample
    total_st_tokens = num_patches * temporal_length
    mask = torch.zeros(total_st_tokens, total_st_tokens, dtype=torch.bool)
    grid_size = int(num_patches ** 0.5)
    #creating a grid iof indices for the patches
    pt_idx_grid = torch.arange(num_patches).view( 1, 1, grid_size, grid_size).float()

    # masking the points which go beyond the image, could be used with query and visibitu too

    points = rearrange(points, 't p d -> p 1 t d').float()
    pt_visibility_mask = rearrange(pt_visibility_mask, 't p -> p t')
    points_in_vis_mask = ((points>-1) & (points<1)).all(dim=-1).squeeze()
    points_in_vis_mask = points_in_vis_mask & pt_visibility_mask
    num_points = points.shape[0]
    pt_idx_grid = pt_idx_grid.repeat(num_points, 1, 1, 1)
    #sampling the patch index in which the points belong to
    sampled_grid = F.grid_sample(pt_idx_grid, points, align_corners=True, mode='nearest')
    sampled_grid = sampled_grid.squeeze()
    # base_sampled_grid is the sampled grid without the frame index
    # TODO(pulkit): Check with shirley about the visibility mask
    base_sampled_grid = sampled_grid.clone().long()
    mask = None
    if return_mask:
        add_points = torch.arange(temporal_length).reshape(1, -1) * num_patches
        # sampled grid patch indices are only spatially indexed, so adding the start index for each frame
        sampled_grid += add_points
        sampled_grid = sampled_grid.long()
        #creating the mask
        for pt_idx in range(num_points):
        #getting the patch indices for the points in the visible mask
            point_ids_to_consider = sampled_grid[pt_idx][points_in_vis_mask[pt_idx]]
            for i, j in combinations(point_ids_to_consider, 2):
                mask[i, j] = True
                #TODO(pulkit): Not sure if both are needed
                mask[j, i] = True
        # set diagnal to 1
        mask = mask + torch.eye(total_st_tokens, dtype=torch.bool)
    
    return mask, base_sampled_grid

def point_sampler(pt_dict, sampler='random', num_points_to_sample=196):

    if sampler == 'random':
        #pred tracks is of shape (t, p, 2)
        if num_points_to_sample > pt_dict['pred_tracks'].shape[1]:
            with_replacement = True
        else:
            with_replacement = False

        point_indices = np.random.choice(np.arange(pt_dict['pred_tracks'].shape[1]),
                                          num_points_to_sample, replace=with_replacement)
    else:
        raise ValueError(f"Sampler {sampler} not supported")
    return point_indices

def process_points(pred_tracks_info, cropped_coords, scale_factor=None):
    
    pred_tracks = pred_tracks_info['pred_tracks'].clone()
    pred_visibility = pred_tracks_info['pred_visibility'].clone()
    per_point_queries = np.argmax(pred_visibility.numpy(), axis=0)
    if scale_factor is not None:
        pred_tracks = pred_tracks * scale_factor
    points_in_cropped = get_valid_points(pred_tracks, per_point_queries,pred_visibility, cropped_coords)
    new_pred_tracks = pred_tracks[:,points_in_cropped]
    new_pred_visibility = pred_visibility[:,points_in_cropped]
    per_point_queries = per_point_queries[points_in_cropped]

    x_min, x_max, y_min, y_max = cropped_coords
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    crop_scale = np.array([crop_width, crop_height]).reshape(1, 1, 2)
    #crop_size is the final crop size
    move_origin = torch.Tensor([x_min, y_min]).view(1, 1, 2)
    new_pred_tracks = new_pred_tracks - move_origin
    new_pred_tracks = (new_pred_tracks / crop_scale)
    # -1 to 1 normalization
    new_pred_tracks = new_pred_tracks * 2 - 1

    new_pred_tracks_info = {'pred_tracks': new_pred_tracks, 
                            'per_point_queries': per_point_queries, 
                            'pred_visibility': new_pred_visibility}
    return new_pred_tracks_info


class PT_Sampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pt_data_dir = cfg.pt_data_dir
        self.dataset = cfg.dataset
    
    def get_pt_scaled_indices(self, decord_num_frames, indices_taken, pred_tracks):
        len_pred_tracks = len(pred_tracks)
        scale_factor = len_pred_tracks / decord_num_frames
        scaled_indices = [int(idx * scale_factor) for idx in indices_taken]
        scaled_indices = [min(idx, len_pred_tracks - 1) for idx in scaled_indices]  # Ensure we don't exceed pred_tracks length
        return scaled_indices

    
    def get_pt_info(self, vid_path, decord_num_frames, indices_taken=None):
        
        if self.dataset == 'k400':
            vid_path = vid_path.replace('_256_new', '') #remove the _256_new from the video name
            split, class_name, vid_name = vid_path.split('/')[-3:]
            class_name = class_name.replace('_', ' ')
            vid_name = vid_name.split('.')[0] + '.pkl'
            vid_name = os.path.join(split, class_name, vid_name)
        else:
            vid_name = vid_path.split('/')[-1].split('.')[0] + '.pkl'
        pt_path = os.path.join(self.pt_data_dir, vid_name)

        with open(pt_path, 'rb') as f:
            pt_data = pickle.load(f)

        pred_tracks = pt_data['pred_tracks'].squeeze()
        pred_visibility = pt_data['pred_visibility'].squeeze()
        if indices_taken is not None:
            pt_indces_to_take = self.get_pt_scaled_indices(decord_num_frames, 
                                                       indices_taken, 
                                                       pred_tracks)
            pred_tracks = pred_tracks[pt_indces_to_take]
            pred_visibility = pred_visibility[pt_indces_to_take]

        #TODO(pulkit): Remove hardcoding of num_patches
        # temp_mask, sampled_grid = self.create_temporal_mask(self.cfg, pred_tracks, pred_visibility, num_patches=196)
        pt_dict = {'pred_tracks': pred_tracks, 
                   'pred_visibility': pred_visibility}

        return pt_dict

    def get_temporal_pt_crops(self,decord_num_frames, pt_dict, temporal_indices, 
                              crop_coords, pt_scale_factor):
        pt_indices_to_take = self.get_pt_scaled_indices(decord_num_frames, 
                                                       temporal_indices, 
                                                       pt_dict['pred_tracks'])
        pt_dict['pred_tracks'] = pt_dict['pred_tracks'][pt_indices_to_take]
        pt_dict['pred_visibility'] = pt_dict['pred_visibility'][pt_indices_to_take]
        pt_dict = process_points(pt_dict, crop_coords, pt_scale_factor)
        return pt_dict
    
    def get_temporal_data(self, pt_dict, num_patches, num_points_to_sample, return_mask=False):
        mask, sampled_grid = get_temporal_connection_data( pt_dict,
                                                           num_patches, 
                                                           return_mask)
        point_indices = point_sampler(pt_dict, sampler='random', 
                                      num_points_to_sample=num_points_to_sample)
        
        pt_visibility = pt_dict['pred_visibility']
        pt_visibility = rearrange(pt_visibility, 't p -> p t')
        pred_tracks = pt_dict['pred_tracks']
        pred_tracks = rearrange(pred_tracks, 't p d -> p t d')
        pred_tracks = pred_tracks[point_indices]
        pt_visibility = pt_visibility[point_indices]
        pt_grid_indices = sampled_grid[point_indices]

        final_pt_dict = {
            'pred_tracks': pred_tracks,
            'pred_grid_indices': pt_grid_indices,
            'pred_visibility': pt_visibility
        }
        if return_mask:
            final_pt_dict['mask'] = mask
        return final_pt_dict

    
    
    
    def process_points_with_crops(self, pt_dict, decord_num_frames,
                                   temporal_indices_per_crop, 
                                   crop_coords, pt_scale_factor):
        sample_pt_dicts = []
        # Create all combinations of temporal indices and crop coordinates
        for temporal_indices, crop_coord in product(temporal_indices_per_crop, 
                                                    crop_coords):
            sample_pt_dict = self.get_temporal_pt_crops(decord_num_frames, pt_dict.copy(), 
                                                temporal_indices, crop_coord, 
                                                pt_scale_factor)
            sample_pt_dicts.append(sample_pt_dict.copy())
        return sample_pt_dicts