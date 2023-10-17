# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import gc
import math
import os
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

import torch
from torch.nn import Module
from tqdm import tqdm

from src.config import FeatureExtractConfig, InferenceConfig, Video, load_model
from src.dataset import create_data_loader_or_dset


@dataclass
class TimeStats:
    """Stores timing statistics for feature extraction."""
    to_load: List[float]
    transfer_device: List[float]
    forward_pass: List[float]
    overall: Optional[float] = None
    to_save: Optional[float] = None


@dataclass
class FeatureExtractionResult:
    """Stores results of feature extraction for a video."""
    result: Dict[str, torch.Tensor]
    time_stats: TimeStats


@dataclass
class ExtractedFeature:
    """Stores a batch of extracted features for a video."""
    video_uid: str
    clip_index: int
    start_time_sec: float
    end_time_sec: float
    feature: Union[List[str], torch.Tensor]
    time_to_load: List[float]
    time_transfer_device: List[float]
    time_forward_pass: List[float]


def num_fvs(vid: Video, config: InferenceConfig) -> int:
    """
    Calculates the expected number of features for a video.

    Used for sanity checking.
    """
    num_frames = vid.frame_count
    fps = vid.frame_rate
    stride_frames = config.stride
    window_size_frames = config.frame_window
    N = num_frames - window_size_frames
    if N < 0:
        return 1
    result = N // stride_frames + 1

    # handle padded frame
    if  N % stride_frames != 0:
        result += 1
    return int(result)


def _extract_features(
    model: Module,
    video: Video,
    config: FeatureExtractConfig,
    max_examples: int = -1,
    silent: bool = False,
) -> Iterator[ExtractedFeature]:
    """
    Extracts features from a video in batches.

    Yields batches of extracted features.
    """
    
    # create dataloader for this "video"
    device = config.inference_config.device
    data = create_data_loader_or_dset(video, config)
    
    for i, x in enumerate(data):
        
        # handle EOF error
        if x == None:
            continue
        
        # v.shape: bs, crop, c, T, w, h 
        v = x["video"]
       
        # for slowfast: [slow_pathway, fast_pathway]
        if isinstance(v, list):
            x["video"] = [i.to(device) for i in v]
            batch_size = v[0].shape[0]
        
        # for other models
        else:
            x["video"] = v.to(device)
            batch_size = v.shape[0]

        with torch.no_grad():
            
            if config.io.debug_mode:
                raise AssertionError("debug mode not implemented")
            else:
                time1 = time.time()
                fv = model(x)
                time2 = time.time()
                time_forward_pass = time2 - time1
                if isinstance(fv, torch.Tensor):
                    fv = fv.detach().cpu()
               
            # return the result for this batch
            yield ExtractedFeature(
                video_uid=x["video_name"],
                clip_index=x["clip_index"],
                start_time_sec=x["clip_start_sec"],
                end_time_sec=x["clip_end_sec"],
                feature=fv,
                time_to_load = -1, # not tracked for now
                time_transfer_device=-1, # not tracked for now
                time_forward_pass= time_forward_pass,
            )

        gc.collect()
        if max_examples > 0 and (i + 1) * batch_size >= max_examples:
            if not silent:
                print("Breaking...")
            break
        t1 = time.time()



def extract_features(
    video: Video,
    config: FeatureExtractConfig,
    model: Optional[Module],
    max_examples: int = -1,
    silent: bool = False,
    assert_feature_size: bool = False,
) -> FeatureExtractionResult:
    """
    Extracts features from a single video.

    Returns extraction results and timing statistics.
    """
   
    
    time_to_load = []
    time_transfer_device = []
    time_forward_pass = []
    
    # calculate the number of "clips" and the number of batches
    batch_size = config.inference_config.batch_size
    total_num_clips = num_fvs(video, config.inference_config)
    batch_num = total_num_clips / max(batch_size, 1)
    batch_num = math.ceil(batch_num)

    # print out stats for this video
    if not silent:
        print(
            f"Extracting features - there are {int(total_num_clips)} for this video.\nThere should be {batch_num} batches.",
            flush=True,
        )

    # this will be run batch_num times
    fvs = list()
    for ef in tqdm(
        _extract_features(
            model, video, config, max_examples=max_examples, silent=silent
        ),
        total = batch_num,
    ):
        
        # append the results for this batch
        assert isinstance(ef.feature, torch.Tensor), "ef.feature is not a tensor"
        ef.feature = ef.feature.cpu()
        fvs.append(ef)
        
        # store time stats
        time_to_load.append(ef.time_to_load)
        time_transfer_device.append(ef.time_transfer_device)
        time_forward_pass.append(ef.time_forward_pass)
    
    
    # sort the features of this video according to time stamp
    fvs.sort(key = lambda x: x.start_time_sec[0])
    assert isinstance(fvs[0].feature, torch.Tensor), "fvs[0].feature is not a tensor"
    
    
    # Stack the results from each batch together
    result = torch.concat([x.feature for x in fvs], dim=0).cpu().detach().squeeze()
    print("final result shape for this video:", result.shape)
        
    # check if the feature number is as expected
    if assert_feature_size:
        fv_amount = result.shape[0]
        expected_fvs = num_fvs( video, config.inference_config)

        # this accounts for rounding error in ffmpeg encoding
        # print("fv_aount: ", fv_amount)
        # print("expected_fvs: ", expected_fvs)
        assert abs(fv_amount - expected_fvs) <= 1
        result = result[:expected_fvs]

    return FeatureExtractionResult(
        result=result,
        time_stats=TimeStats(
            to_load=time_to_load,
            transfer_device=time_transfer_device,
            forward_pass=time_forward_pass,
        ),
    )


def perform_feature_extraction(
    videos: List[Video], config: FeatureExtractConfig
) -> TimeStats:
    """
    Main function to extract features from multiple videos.

    Returns overall timing statistics.
    """
    
    os.makedirs(config.io.out_path, exist_ok=True)

    # Get the stats for the entire extraction
    time_stats = TimeStats(
        to_load=[],
        transfer_device=[],
        forward_pass=[],
        overall=0,
        to_save=0,
    )
    
    # sort by smallest video first
    print(f"Number of videos = {len(videos)}")
    videos.sort(key=lambda x: x.frame_count)
    model = load_model(config)
    o1 = time.time()
    
    # Extract feature from each "video" one at a time
    for vid in tqdm(videos, desc="videos"):
        
        print("")
        print("")
        print("")
        print("Video: ", vid.uid)
        gc.collect()
        
        # Extract feature from a "video"
        feature_extract_result = extract_features(
            video = vid,
            config = config,
            model = model,
            assert_feature_size= True,
        )
        result = feature_extract_result.result

        # Save feature for a single "video"
        t1 = time.time()
        torch.save(result, f"{config.io.out_path}/{vid.uid}.pt")
        t2 = time.time()

        # Time stats for a "video"
        time_stats.to_save += t2 - t1
        time_stats.to_load.extend(feature_extract_result.time_stats.to_load)
        time_stats.transfer_device.extend(
            feature_extract_result.time_stats.transfer_device
        )
        time_stats.forward_pass.extend(feature_extract_result.time_stats.forward_pass)
        
    o2 = time.time()
    time_stats.overall = o2 - o1
    return time_stats
