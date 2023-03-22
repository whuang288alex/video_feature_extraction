# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from fractions import Fraction
from typing import Any, List

import av
import numpy as np
import torch
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config import FeatureExtractConfig, Video, get_transform

import warnings
warnings.filterwarnings("ignore")
    
# Encode a single "video" and store it. Only decode it when get_frames is called (once for each "clip")
class EncodedVideoCached:
    
    def __init__(self, path, frame_buffer_size=100):
        self.path = path
        self.vid = EncodedVideo.from_path(path, decoder="pyav")
        self.vid._container.seek(0)
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        self.last_t = None

    # this function is used to get a "clip" from the encoded video based on start_time and end_time
    def get_clip(self, t1, t2):
        if self.last_t is not None and t1 < self.last_t:
            raise AssertionError("cannot seek backward")
        
        vstream = self.vid._container.streams.video[0]
        vs = vstream.start_time * vstream.time_base
    
        frames = self.get_frames(
            self.vid._container,
            t1 + vs,
            t2 + vs,
            self.frame_buffer,
            self.frame_buffer_size,
        )
        
        self.last_t = t1
        return {
            "num_frames": len(frames),
            "video": thwc_to_cthw(
                torch.stack(
                    [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in frames]
                )
            ).to(torch.float32),
            "audio": None,
        }
        
    # this function is used to get "frames" for a "clip" based on start_time and end_time
    def get_frames(self, container, t1, t2, buffer, max_buffer_size):
        ret = []
        tb = container.streams.video[0].time_base

        def is_in_range(frame):
            t = frame.pts * tb
            return t >= t1 and t < t2

        def exceeds_range(frame):
            return frame.pts * tb >= t2

        for frame in buffer:
            if is_in_range(frame):
                ret.append(frame)
        
        prev_pts = None
        
        # This try except block is to avoid the EOF error that arrives because t2 exceeds the frame range 
        try:
            for frame in container.decode(video=0):
                if frame.pts is None:
                    raise AssertionError("frame is None")
                if prev_pts is not None and frame.pts < prev_pts:
                    raise AssertionError("failed assumption pts in order: ")
                if not isinstance(frame, av.VideoFrame):
                    raise AssertionError("other packets not supported")
                prev_pts = frame.pts
                buffer.append(frame)
                if len(buffer) > max_buffer_size:
                    del buffer[0]

                if is_in_range(frame):
                    ret.append(frame)
                elif exceeds_range(frame):
                    break
        except:
            pass
    
        pts_in_ret = [frame.pts for frame in ret]
        if not (np.diff(pts_in_ret) > 0).all():
            raise AssertionError("not increasing sequence of frames")
        return ret
  
    @property
    def duration(self) -> float:
        vstream = self.vid._container.streams.video[0]
        return vstream.duration * vstream.time_base

# Returns a dataset for a single "video". Each entry corresponds to a "clip"
class IndexableVideoDataset(torch.utils.data.Dataset):

    def __init__(
        self, config: FeatureExtractConfig, video: Video, sampler, transform
    ):
        assert (
            config.inference_config.include_audio
            ^ config.inference_config.include_video
        ), "cannot extract features from both audio and video at the same time"
        
        """
        cannot include audio and video at the same time
        """
        self.config = config
        self.sampler = sampler
        self.transform = transform

        if self.config.inference_config.include_video:
            self.encoded_videos = EncodedVideoCached(video.path, 2 * config.inference_config.frame_window)
        else:
            raise AssertionError("Audio not implemented")

        self.clips = list(self.get_all_clips(video, self.encoded_videos.duration, sampler))

    # this function calculate the timestamp for each clip according to the sampler
    def get_all_clips(self, video, video_length, sampler):
        last_clip_time = 0.0
        annotation = {}
        n_clips = 0
        while True:
            clip = sampler(last_clip_time, video_length, annotation)
            last_clip_time = clip.clip_end_sec
            n_clips += 1
            yield (video, clip)
            if clip.is_last_clip:
                break

    def __len__(self):
        return len(self.clips)

    # returns a single "clip"
    def __getitem__(self, idx):
        video, clip = self.clips[idx]
        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            is_last_clip,
        ) = clip

        encoded_video = self.encoded_videos # this is the EncodedVideoCached instance
        datum = encoded_video.get_clip(clip_start, clip_end)

        # if this clip does not have enough frame, pad it with zeros
        if datum['num_frames'] < self.config.inference_config.frame_window:
            pad = (self.config.inference_config.frame_window - datum['num_frames'])
            datum["video"] = torch.cat([datum["video"], torch.zeros(3, pad, *datum["video"].shape[2:])], dim = 1)
            datum['num_frames'] = datum["video"].shape[1]

        # if this clip has too many frames, only get frame_window frames
        elif datum['num_frames'] > self.config.inference_config.frame_window:
            datum["video"] = datum["video"][:, :self.config.inference_config.frame_window]
            datum['num_frames'] = datum["video"].shape[1]
    
        # force checking the number of frames to guard against missing frames
        assert datum['num_frames'] == self.config.inference_config.frame_window
        
        # the info for this clip
        sample_dict = {
            "video_name": video.uid,
            "video_index": idx,
            "clip_index": clip_index,
            "aug_index": aug_index,
            "is_stereo": video.is_stereo,
            "clip_start_sec": float(clip_start),
            "clip_end_sec": float(clip_end),
        }
        
        if datum["video"] is not None:
            sample_dict["video"] = datum["video"]
        else:
            raise AssertionError("Audio not implemented")
        
        # apply transform at the "clip" level
        sample_dict = self.transform(sample_dict)
        return sample_dict
    
def create_dset(
    video: Video, config: FeatureExtractConfig
) -> IndexableVideoDataset:
    
    if config.io.debug_mode:
        raise AssertionError("debug mode not implemented")
    
    # this is used to get the time stamps for each "clip"
    clip_sampler = UniformClipSampler(
        
        # how many seconds each clip is in 
        clip_duration=Fraction(
            config.inference_config.frame_window,  Fraction(video.frame_rate)
        )
        if isinstance(config.inference_config.frame_window, int)
        else config.inference_config.frame_window,
        
        # how many seconds each stride is in
        stride = Fraction(config.inference_config.stride,  Fraction(video.frame_rate))
        if isinstance(config.inference_config.stride, int)
        else config.inference_config.stride,
        backpad_last=True,
    )
    
    # return a custom dataset
    return IndexableVideoDataset(
        config, video, clip_sampler, Compose([get_transform(config),])
    )


def create_data_loader(dset, config: FeatureExtractConfig) -> DataLoader:
    
    if config.inference_config.batch_size == 0:
        raise AssertionError("batch size zero is not supported") 
    
    return DataLoader(
        dset,
        batch_size=config.inference_config.batch_size,
        num_workers=config.inference_config.num_workers,
        prefetch_factor=config.inference_config.prefetch_factor
    )


def create_data_loader_or_dset(
    video: Video, config: FeatureExtractConfig
) -> Any:
    dset = create_dset(video, config)
    return create_data_loader(dset=dset, config=config)
