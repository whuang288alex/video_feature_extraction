#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os, sys
sys.path.insert(0, os.getcwd())

import argparse
import datetime
import functools
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import hydra

import numpy as np
import submitit
import torch
from config import (
    FeatureExtractConfig,
    get_videos,
    ScheduleConfig,
    Video,
)
from extract_features import num_fvs, perform_feature_extraction
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm

def print_stats_for_videos(
    config: FeatureExtractConfig, all_videos: List[Video], videos: List[Video]
):
    # stats for uids
    assert isinstance(all_videos[0], Video)
    assert isinstance(videos[0], Video)
    total_secs_uncompleted = sum(v.frame_count * config.fps for v in videos)
    secs_uncompleted = sum(v.frame_count * config.fps for v in all_videos)

    print(
        f"""
    Total Number of Videos = {len(all_videos)}
    Incomplete videos = {len(videos)}

    Total Seconds = {total_secs_uncompleted}
    Incomplete seconds = {secs_uncompleted} = {secs_uncompleted/total_secs_uncompleted * 100:.2f}%
    """
    )

def print_completion_stats(results):
    time_to_load = []
    time_to_transfer = []
    forward_pass_time = []
    print(
        "overall,save_time,avg_total,avg_load_time,avg_transfer_time,avg_forward_pass"
    )
    for result in results:
        time_to_load.extend(result.to_load)
        time_to_transfer.extend(result.transfer_device)
        forward_pass_time.extend(result.forward_pass)

        ttl = torch.Tensor(result.to_load)
        ttt = torch.Tensor(result.transfer_device)
        fpt = torch.Tensor(result.forward_pass)
        mean_sum = ttl.mean() + ttt.mean() + fpt.mean()
        print(
            f"{result.overall},{result.to_save},{mean_sum},{ttl.mean()},{ttt.mean()},{fpt.mean()}"
        )

    print("")
    print("")

    print("Averages")
    print("mean_forward, only_forward_pass, time_to_load,time_to_transfer")
    ttl = torch.Tensor(time_to_load)
    ttt = torch.Tensor(time_to_transfer)
    fpt = torch.Tensor(forward_pass_time)
    mean_sum = ttl.mean() + ttt.mean() + fpt.mean()
    print(f"{mean_sum},{ttl.mean()},{ttt.mean()},{fpt.mean()}")


@hydra.main(config_path="configs", config_name=None)
def schedule_feature_extraction(config: FeatureExtractConfig):
    
    # make file path relative
    config.io.video_dir_path = os.path.dirname(os.path.abspath(__file__)) + config.io.video_dir_path
    config.io.ego4d_download_dir = os.path.dirname(os.path.abspath(__file__)) + config.io.ego4d_download_dir
    config.io.out_path = os.path.dirname(os.path.abspath(__file__)) + config.io.out_path
    config.io.debug_path = os.path.dirname(os.path.abspath(__file__)) + config.io.debug_path
    os.makedirs(config.io.out_path, exist_ok=True)
    
    print("###################### Feature Extraction Config ####################")
    print(OmegaConf.to_yaml(config))
    print("############################################################")

    # load "ALL VIDEOS"
    videos, all_videos = get_videos(config)
    
    # generate a config file for "EACH EXTRACTION"
    with open(f"{config.io.out_path}/config.yaml", "w") as out_f:
        out_f.write(OmegaConf.to_yaml(config))
    if len(videos) == 0:
        return
    
    # print stats for  "ALL VIDEOS"
    print_stats_for_videos(config, all_videos=all_videos, videos=videos)

    # get results for "ALL VIDEOS"
    results = perform_feature_extraction(videos, config)
    
    # print stats for "ALL VIDEOS"
    print_completion_stats([results])

if __name__ == "__main__":
    schedule_feature_extraction()
