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
import torchvision
from torch.nn import Module
from tqdm import tqdm

from config import FeatureExtractConfig, InferenceConfig, Video, load_model
from dataset import create_data_loader_or_dset


@dataclass
class TimeStats:
    to_load: List[float]
    transfer_device: List[float]
    forward_pass: List[float]
    overall: Optional[float] = None
    to_save: Optional[float] = None


@dataclass
class FeatureExtractionResult:
    result: Dict[str, torch.Tensor]
    time_stats: TimeStats


@dataclass
class ExtractedFeature:
    video_uid: str
    clip_index: int
    start_time_sec: float
    end_time_sec: float
    feature: Union[List[str], torch.Tensor]
    time_to_load: List[float]
    time_transfer_device: List[float]
    time_forward_pass: List[float]


# calculate expected number of feature number according to frame number
def num_fvs(vid: Video, config: InferenceConfig) -> int:
    def _num_fvs(
        num_frames: int,
        fps: float,
        stride_frames: int,
        window_size_frames: int,
        backpad_last: bool = True,
    ) -> int:
        N = num_frames - window_size_frames
        if N < 0:
            return 1
        result = N // stride_frames + 1

        # handle padded frame
        if backpad_last and N % stride_frames != 0:
            result += 1
        return result
    
    return _num_fvs(vid.frame_count, vid.frame_rate, config.stride, config.frame_window)


# returns extraction results from a batch of "CLIPS"
def _extract_features(
    model: Module,
    video: Video,
    config: FeatureExtractConfig,
    max_examples: int = -1,
    silent: bool = False,
) -> Iterator[ExtractedFeature]:
    
    # NOTE: create_data_loader_or_dset actually only creates dataset that contains one video now
    device = config.inference_config.device
    data = create_data_loader_or_dset([video], config)
    t1 = time.time()
    
    # x["video"].shape: bs, crop, c, T, w, h (x is clips for a batch)
    for i, x in enumerate(data):
        
        # handle EOF error
        if x == None:
            continue
        
        t2 = time.time()
        load_time = t2 - t1
        t1 = time.time()

        # avoid "Too many open files" error
        # x_cp = deepcopy(x)
        # del x
        # x = x_cp
        v = x["video"]
    
        
        # for slowfast: ([slow_pathway, fast_pathway])
        if isinstance(v, list):
            x["video"] = [i.to(device) for i in v]
            batch_size = v[0].shape[0]
        # for other models
        else:
            x["video"] = v.to(device)
            batch_size = v.shape[0]
         
                
        t2 = time.time()
        transfer_time = t2 - t1
        t1 = time.time()

    
        with torch.no_grad():
            
            # for normal feature extraction
            if not config.io.debug_mode:
                fv = model(x)
                if isinstance(fv, torch.Tensor):
                    fv = fv.detach().cpu()
            
            # only for debug purpose
            else:
                vid_inp = x["video"]
                fv = None
                for batch_idx, (video_name, clip_index) in enumerate(
                    zip(x["video_name"], x["clip_index"])
                ):
                    to_dir = f"{config.io.debug_path}/{video_name}"
                    to_path = f"{to_dir}/{clip_index}.jpg"
                    os.makedirs(to_dir, exist_ok=True)
                    grid = torchvision.utils.make_grid(
                        vid_inp[batch_idx].permute(1, 0, 2, 3),
                        nrows=vid_inp.shape[2] // config.inference_config.stride,
                    )
                    torchvision.utils.save_image(grid / 255.0, fp=to_path)  # noqa
            
            t2 = time.time()
            forward_pass_time = t2 - t1
            # return the result for this batch
            # clip_index: clips that are in this batch, should have length = batch_size
            yield ExtractedFeature(
                video_uid=x["video_name"],
                clip_index=x["clip_index"],
                start_time_sec=x["clip_start_sec"],
                end_time_sec=x["clip_end_sec"],
                feature=fv,
                time_to_load=load_time,
                time_transfer_device=transfer_time,
                time_forward_pass=forward_pass_time,
            )

        gc.collect()
        if max_examples > 0 and (i + 1) * batch_size >= max_examples:
            if not silent:
                print("Breaking...")
            break
        t1 = time.time()


# returns extraction results from "A VIDEO"
def extract_features(
    video: Video,
    config: FeatureExtractConfig,
    model: Optional[Module] = None,
    log_info: bool = True,
    max_examples: int = -1,
    silent: bool = False,
    assert_feature_size: bool = False,
) -> FeatureExtractionResult:

    if model is None:
        model = load_model(config)
    
    time_to_load = []
    time_transfer_device = []
    time_forward_pass = []
    
    # calculate the number of clips and the number of batches
    batch_size = config.inference_config.batch_size
    total_num_clips = num_fvs(video, config.inference_config)
    batch_num = total_num_clips / max(batch_size, 1)
    batch_num = math.ceil(batch_num)

    # print out stats for this video
    if not silent:
        print(
            f"Extracting features - there are {total_num_clips} for this video.\nThere should be {batch_num} batches.",
            flush=True,
        )

    # this will run batch_num times
    fvs = list()
    for ef in tqdm(
        _extract_features(
            model, video, config, max_examples=max_examples, silent=silent
        ),
        total = batch_num,
    ):
        if config.io.debug_mode:
            continue

        if isinstance(ef.feature, torch.Tensor):
            ef.feature = ef.feature.cpu()
        fvs.append(ef)
        
        # store time stats
        time_to_load.append(ef.time_to_load)
        time_transfer_device.append(ef.time_transfer_device)
        time_forward_pass.append(ef.time_forward_pass)
    
    
    # sort the features of this video according to time stamp
    fvs.sort(key = lambda x: x.start_time_sec[0])


    # for normal feature extraction
    if isinstance(fvs[0].feature, torch.Tensor):
        
        # Stack the results from each batch together
        result = (
            torch.concat([x.feature for x in fvs], dim=0)
            .cpu()
            .detach()
            .squeeze()
        )  
        print("final result shape for this video:", result.shape)
    
    # in debug mode
    else:
        result = [
            {
                "start_time_sec": x.start_time_sec.item(),
                "end_time_sec": x.end_time_sec.item(),
                "feature": x.feature,
            }
            for x in fvs
            if x.feature is not None
        ]
        
    # check if the feature number is as expected
    fv_amount = result.shape[0]
    expected_fvs = num_fvs( video, config.inference_config)
    
    if expected_fvs != fv_amount:
        if assert_feature_size:
            # this accounts for rounding error in ffmpeg encoding
            print("fv_aount: ", fv_amount)
            print("expected_fvs: ", expected_fvs)
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
    
    is_audio_model = config.inference_config.include_audio
    o1 = time.time()
    
    # Extract feature from "EACH VIDEO" one at a time
    for vid in tqdm(videos, desc="videos"):
        
        print("")
        print("")
        print("")
        print(vid.uid)
        gc.collect()
        
        # Extract feature from "A VIDEO"
        feature_extract_result = extract_features(
            vid,
            config,
            assert_feature_size=not is_audio_model,
        )
        result = feature_extract_result.result

        # Save feature for "A VIDEO"
        t1 = time.time()
        torch.save(result, f"{config.io.out_path}/{vid.uid}.pt")
        t2 = time.time()

        # Time stats for "A VIDEO"
        time_stats.to_save += t2 - t1
        time_stats.to_load.extend(feature_extract_result.time_stats.to_load)
        time_stats.transfer_device.extend(
            feature_extract_result.time_stats.transfer_device
        )
        time_stats.forward_pass.extend(feature_extract_result.time_stats.forward_pass)
        
    o2 = time.time()
    time_stats.overall = o2 - o1
    return time_stats
