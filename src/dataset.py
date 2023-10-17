# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
from fractions import Fraction
from typing import Any, List
import av
import numpy as np
import torch
import math
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.utils.data as tdata
from src.config import FeatureExtractConfig, Video, get_transform

import warnings
warnings.filterwarnings("ignore")
    

class EncodedVideoCached:
    """
    Encodes and caches a video to avoid repeatedly decoding frames.
    Only decodes frames when get_frames() is called for each clip.
    """
    
    def __init__(self, path, frame_buffer_size):
        self.path = path
        self.vid = EncodedVideo.from_path(path, decoder="pyav")
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        self.last_t = None


    def set_seek(self, worker_id, num_workers):
        """
        Seeks to the appropriate position in the video for each worker.
        """
        self.vid._container.seek(min(self.vid._container.duration//num_workers * worker_id - 100, 0))
    
    
    def get_clip(self, t1, t2, is_last_clip = False):
        """
        Gets a clip from the encoded video based on start/end times.

        Args:
            t1 (float): Start time of clip in seconds
            t2 (float): End time of clip in seconds
            is_last_clip (bool): Whether this is the last clip

        Returns:
            dict: Contains decoded frames, number of frames, etc.
        """
        if self.last_t is not None and t1 < self.last_t:
            raise AssertionError("cannot seek backward")
        
        vstream = self.vid._container.streams.video[0]
        vs = (vstream.start_time)* vstream.time_base
    
        frames = self.get_frames(
            self.vid._container,
            t1 + vs,
            t2 + vs,
            self.frame_buffer,
            self.frame_buffer_size,
        )
        
        self.last_t = t1
        try:
            ret = {
                "num_frames": len(frames),
                "video": thwc_to_cthw(
                    torch.stack(
                        [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in frames]
                    )
                ).to(torch.float32),
            }
        except Exception as e:
            if is_last_clip:
                # This is the last clip, so we can ignore this error because there might be some rounding error in terms of number of clips in the video
                return None
        return ret
        
        
    def get_frames(self, container, t1, t2, buffer, max_buffer_size):
        """
        Decodes frames for a clip based on start/end times.
        
        Args:
            container: Video container
            t1 (float): Start time of clip in seconds
            t2 (float): End time of clip in seconds
            buffer (list): Buffer of previously decoded frames
            max_buffer_size (int): Max number of frames to store in buffer

        Returns:
            list: List of decoded video frames for the specified clip
        """
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
                    raise AssertionError("[ERROR] Frame is None")
                if prev_pts is not None and frame.pts < prev_pts:
                    raise AssertionError("[ERROR] Failed assumption pts in order")
               
                prev_pts = frame.pts
                buffer.append(frame)
                if len(buffer) > max_buffer_size:
                    del buffer[0]

                if is_in_range(frame):
                    ret.append(frame)
                elif exceeds_range(frame):
                    break
        except EOFError as e:
            pass
    
        pts_in_ret = [frame.pts for frame in ret]
        if not (np.diff(pts_in_ret) > 0).all():
            raise AssertionError("not increasing sequence of frames")
        return ret
  
    @property
    def duration(self) -> float:
        """
        Returns:
            float: Duration of video in seconds
        """
        vstream = self.vid._container.streams.video[0]
        return vstream.duration * vstream.time_base


class IterableVideoDataset(torch.utils.data.IterableDataset):
    """
    This class is used to create a custom iterable dataset that is efficient for loading and decoding videos
    """
    
    def __init__(self, config, video, sampler, transform):
        self.config = config
        self.sampler = sampler
        self.transform = transform
        self.encoded_videos = EncodedVideoCached(video.path, 2 * config.inference_config.frame_window)
        self.clips_info = list(self.get_all_clips_info(video, self.encoded_videos.duration, sampler))

    
    def get_all_clips_info(self, video, video_length, sampler):
        """
        Generates metadata for all clips in the video using the sampler.

        Runs only once across all workers.
        """
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
    
    def __iter__(self): 
        """
        Generates metadata for all clips in the video using the sampler.

        Runs only once across all workers.
        """
        
        # get worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info == None:
            worker_id, num_workers = 0, 1 # single process loading
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        
        
        # divide the workload and set the seek for each worker
        workload_per_worker = math.ceil((len(self.clips_info))/float(num_workers))
        self.iter_start = worker_id * workload_per_worker
        self.iter_end = min(self.iter_start + workload_per_worker, len(self.clips_info))
        self.encoded_videos.set_seek(worker_id, num_workers)
            
            
        # return iterator
        for i in range(self.iter_start, self.iter_end): 
            
            video, (clip_start, clip_end, clip_index, aug_index, is_last_clip) = self.clips_info[i]
            clip = self.encoded_videos.get_clip(clip_start, clip_end, is_last_clip)
            
            # If this is the last clip, we can ignore the error because there might be some rounding error in terms of number of clips in the video
            if clip is None:
                clip = {
                    "num_frames": self.config.inference_config.frame_window,
                    "video": torch.zeros(3, self.config.inference_config.frame_window, 180, 320),   # size of thumos dataset frame is 180 * 320
                    "audio": None,
                } 
            # if this clip does not have enough frame, pad it with zeros
            elif clip['num_frames'] < self.config.inference_config.frame_window:
                pad = (self.config.inference_config.frame_window - clip['num_frames'])
                clip["video"] = torch.cat([clip["video"], torch.zeros(3, pad, * clip["video"].shape[2:])], dim = 1)
                clip['num_frames'] = clip["video"].shape[1]

            # if this clip has too many frames, only get frame_window frames
            elif clip['num_frames'] > self.config.inference_config.frame_window:
                clip["video"] = clip["video"][:, :self.config.inference_config.frame_window]
                clip['num_frames'] = clip["video"].shape[1]

            
            # the info for this clip
            data_dict = {
                "video_name": video.uid,
                "video_index": i,
                "video": clip["video"],
                "clip_index": clip_index,
                "aug_index": aug_index,
                "is_stereo": video.is_stereo,
                "clip_start_sec": float(clip_start),
                "clip_end_sec": float(clip_end),
            }
            
            # apply transform at the "clip" level
            data_dict = self.transform(data_dict)
            yield data_dict
    

def create_dset(
    video: Video, config: FeatureExtractConfig
) -> IterableVideoDataset:
    """
    Creates the iterable video dataset.

    Args:
        video (Video): Video metadata
        config (FeatureExtractConfig): Configuration

    Returns:
        IterableVideoDataset: Iterable video dataset
    """
    
    # this is used to get the time stamps for each "clip"
    clip_sampler = UniformClipSampler(
        # how many seconds each clip is in 
        clip_duration=Fraction(
            config.inference_config.frame_window, Fraction(video.frame_rate)
        )
        if isinstance(config.inference_config.frame_window, int)
        else config.inference_config.frame_window,
        
        # how many seconds each stride is in
        stride = Fraction(config.inference_config.stride,  Fraction(video.frame_rate))
        if isinstance(config.inference_config.stride, int)
        else config.inference_config.stride,
        backpad_last=True,
    )
    
    # return a custom interable dataset
    return IterableVideoDataset(
        config, video, clip_sampler, Compose([get_transform(config),])
    )


def create_data_loader(dset, config: FeatureExtractConfig) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the video dataset.

    Args:
        dset: Video dataset
        config (FeatureExtractConfig): Configuration

    Returns:
        DataLoader: PyTorch DataLoader for video dataset
    """
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
    """
    Creates either a DataLoader or just the dataset based on config.

    Args:
        video (Video): Video metadata
        config (FeatureExtractConfig): Configuration

    Returns:
        DataLoader or IterableVideoDataset
    """
    dset = create_dset(video, config)
    return create_data_loader(dset=dset, config=config)
