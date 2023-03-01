#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys

from typing import List

import hydra
import torch
from omegaconf import OmegaConf
from get_videos import get_videos
from config import FeatureExtractConfig, Video
from extract_features import perform_feature_extraction


sys.path.insert(0, os.getcwd())
root =  os.path.dirname(os.path.abspath(__file__))

def print_stats_for_videos(
    config: FeatureExtractConfig, all_videos: List[Video], videos: List[Video]
):
    # stats for uids
    assert isinstance(all_videos[0], Video)
    assert isinstance(videos[0], Video)

    print(
        f"""
    Total Number of Videos = {len(all_videos)}
    Incomplete videos = {len(videos)}
    """
    )

def print_completion_stats(results):
    time_to_load = []
    time_to_transfer = []
    forward_pass_time = []
    
    print("")
    print("")
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
    config.io.video_dir_path = os.path.join (root, config.io.video_dir_path)
    assert os.path.exists(config.io.video_dir_path),  "The video path provided in the config file does not exist."
    
    config.io.out_path = os.path.join (root, config.io.out_path)
    os.makedirs(config.io.out_path, exist_ok=True)
    
    config.io.ego4d_download_dir = os.path.join (root, config.io.ego4d_download_dir)
    config.io.debug_path = os.path.join (root, config.io.debug_path)
    
    print("###################### Feature Extraction Config ####################")
    print(OmegaConf.to_yaml(config))
    print("############################################################")

    # load all the "videos" in the specified directory
    videos, all_videos = get_videos(config)
    
    with open(f"{config.io.out_path}/config.yaml", "w") as out_f:
        out_f.write(OmegaConf.to_yaml(config))
    
    if len(videos) == 0:
        return
    
    print_stats_for_videos(config, all_videos=all_videos, videos=videos)
    results = perform_feature_extraction(videos, config)
    print_completion_stats([results])


if __name__ == "__main__":
    schedule_feature_extraction()
