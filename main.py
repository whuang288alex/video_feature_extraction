#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys
import hydra
import torch
from typing import List
from omegaconf import OmegaConf
from src.video_info import get_videos
from src.config import FeatureExtractConfig, Video
from src.feature_extraction import perform_feature_extraction

def print_stats_for_videos(
    all_videos: List[Video], videos: List[Video]
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

@hydra.main(config_path="configs", config_name=None)
def schedule_feature_extraction(config: FeatureExtractConfig):
    
    print("###################### Feature Extraction Config ####################")
    print(OmegaConf.to_yaml(config))
    print("############################################################")
    assert os.path.exists(config.io.video_dir_path),  "The video path provided in the config file does not exist."
    os.makedirs(config.io.out_path, exist_ok=True)

    
    # load all the videos in the specified directory and filter out the ones that have already been processed
    videos, all_videos = get_videos(config)
    assert len(videos) > 0, "No videos found in the video directory or all videos have already been processed."
    print_stats_for_videos(all_videos=all_videos, videos=videos)
    
    
    # perform feature extraction and save the config file
    perform_feature_extraction(videos, config)
    with open(f"{config.io.out_path}/config.yaml", "w") as out_f:
        out_f.write(OmegaConf.to_yaml(config))
        

if __name__ == "__main__":
    schedule_feature_extraction()
