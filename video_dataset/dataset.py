#!/usr/bin/env python

import os, sys
from typing import Optional
import av
import io
import numpy as np
try:
    from decord import VideoReader
    DECORD_AVAILABLE = True
    print("DECORD_AVAILABLE")
except ImportError:
    DECORD_AVAILABLE = False

import torch
from torchvision import transforms

from .transform import create_random_augment, random_resized_crop
from einops import rearrange

class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self, list_path: str, data_root: str,
        num_spatial_views: int, num_temporal_views: int, random_sample: bool,
        num_frames: int, sampling_rate: int, spatial_size: int,
        mean: torch.Tensor, std: torch.Tensor,
        auto_augment: Optional[str] = None, interpolation: str = 'bicubic',
        mirror: bool = False, cfg: Optional[dict] = None,
    ):
        self.data_root = data_root
        self.interpolation = interpolation
        self.spatial_size = spatial_size

        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.auto_augment = auto_augment
        else:
            assert auto_augment is None and not mirror
            self.random_sample = False
            self.num_temporal_views = num_temporal_views
            self.num_spatial_views = num_spatial_views

        with open(list_path) as f:
            self.data_list = f.read().splitlines()
        self.cfg = cfg
        self.vid_base_dir = cfg.vid_base_dir
        if 'SCRATCH_DIR' in os.environ and 'SCRATCH_DIR' in self.vid_base_dir:
            self.vid_base_dir  = self.vid_base_dir.replace('$SCRATCH_DIR', os.environ['SCRATCH_DIR'])
        if cfg.dataset == 'ssv2':
            self.vid_base_dir = os.path.join(self.vid_base_dir, 
                                             '20bn-something-something-v2')

        self.total_views = num_spatial_views * num_temporal_views
        


    def __len__(self):
        return len(self.data_list)
    



    

    def __getitem__(self, idx):
        line = self.data_list[idx]
        path, label = line.split(' ')
        path = os.path.join(self.data_root, path)
        label = int(label)

        path = os.path.join(self.vid_base_dir, path)
        if DECORD_AVAILABLE:
            frames = VideoReader(path, num_threads=1)
        else:
            frames = {}
            container = av.open(path)
            for frame in container.decode(video=0):
                frames[frame.pts] = frame
            container.close()
            frames = [frames[k] for k in sorted(frames.keys())]

        if self.random_sample:
            frame_idx = self._random_sample_frame_idx(len(frames))
            if DECORD_AVAILABLE:
                frames = frames.get_batch(frame_idx).asnumpy()
                frames = torch.as_tensor(frames / 255.).float()
            else:
                frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
                frames = torch.as_tensor(np.stack(frames)).float() / 255.

            if self.auto_augment is not None:
                aug_transform = create_random_augment(
                    input_size=(frames.size(1), frames.size(2)),
                    auto_augment=self.auto_augment,
                    interpolation=self.interpolation,
                )
                frames = frames.permute(0, 3, 1, 2) # T, C, H, W
                frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
                frames = aug_transform(frames)

                frames = torch.stack([transforms.ToTensor()(img) for img in frames])
                frames = frames.permute(0, 2, 3, 1)

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            frames = random_resized_crop(
                frames, self.spatial_size, self.spatial_size,
            )
            
        else:
            frames = self._generate_temporal_crops_idx_with_crops(frames)
            # frames = [x.to_rgb().to_ndarray() for x in frames]
            frames = torch.as_tensor(np.stack(frames))
            num_crops, temporal_len, _, _, _ = frames.size()
        
            if num_crops > 1:
                frames = rearrange(frames, 'n t h w -> (n t) h w c')
            else:
                frames = frames.squeeze(0)

            frames = frames.float() / 255.

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            
            if frames.size(-2) < frames.size(-1):
                new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                new_height = self.spatial_size
            else:
                new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                new_width = self.spatial_size
            frames = torch.nn.functional.interpolate(
                frames, size=(new_height, new_width),
                mode='bilinear', align_corners=False,
            )

            frames = self._generate_spatial_crops(frames)
            # frames = sum([self._generate_temporal_crops(x) for x in frames], [])

            frames = torch.stack(frames)
            if num_crops > 1:
                frames = rearrange(frames, '(n t) h w c -> n t h w c', n=num_crops)

        return frames, label


    def _generate_temporal_crops(self, frames):
        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        if frames.size(1) < seg_len:
            frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
        slide_len = frames.size(1) - seg_len

        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        
        return crops


    def _generate_temporal_crops_idx_with_crops(self, frames):
        # assumption is that frames is a decord VideoReader object

        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        all_frame_indices = list(range(len(frames)))
        if len(frames) < seg_len:
            all_frame_indices = all_frame_indices + [all_frame_indices[-1]] * (seg_len - len(frames))
            
        slide_len = len(all_frame_indices) - seg_len

        crop_indices = []
        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crop_indices.append(all_frame_indices[st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        for crop_idx in crop_indices:
            crops.append(frames.get_batch(crop_idx).asnumpy())

        return crops
        


    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops
        
        else:
            raise NotImplementedError()


    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, list_path: str, num_frames: int, num_views: int, spatial_size: int):
        with open(list_path) as f:
            self.len = len(f.read().splitlines())
        self.num_frames = num_frames
        self.num_views = num_views
        self.spatial_size = spatial_size

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
        if self.num_views != 1:
            shape = [self.num_views] + shape
        return torch.zeros(shape), 0
