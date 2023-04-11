# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math

import cv2
from config import FeatureExtractConfig, InputOutputConfig, Video


# helper function to get all the videos in the input directory
def _videos(config: InputOutputConfig, unfiltered: bool = False) -> List[Video]:
    
    
    # list all the files in the directory
    def _uids_for_dir(path: str) -> List[str]:
        root =  os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(root, "video_list.txt")) as file:
            video_list = [f"{path}/{line.rstrip()}"  for line in file]
            
        ret = [
            p
            for p in os.listdir(path)
            if p in video_list
        ]
        return [Path(p).stem for p in ret]
    
    
    # returns uids of all the videos
    def _unfiltered_uids(config: InputOutputConfig) -> List[str]:
        uids = config.uid_list
        if uids is None:
            assert config.video_dir_path is not None, "Not given any uids"
            uids = _uids_for_dir(config.video_dir_path)
        return uids


    # only returns uids of videos that are not yet processed
    def _uids(config: InputOutputConfig) -> List[str]:
        uids = _unfiltered_uids(config)
        if config.filter_completed:
            completed_uids = set(_uids_for_dir(config.out_path))
            uids = [uid for uid in uids if uid not in completed_uids]

        assert uids is not None, "`uids` is None"
        assert len(uids) >= 0, "`len(uids)` is 0"
        return uids

    
    # construct absolute file path
    def _path_for(config: InputOutputConfig, uid: str) -> str:
        return f"{config.video_dir_path}/{uid}.mp4"


    # calculate video infos for this video
    def get_frame_count(video_path):
        vid = cv2.VideoCapture(video_path)
        length = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        # print("frame count: ", length)
        return length
    
    def get_frame_rate(video_path):
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        # print("fps: ", fps)
        return fps
    
    def get_w(video_path):
        vid = cv2.VideoCapture(video_path)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        return width
    
    def get_h(video_path):
        vid = cv2.VideoCapture(video_path)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return height

    uids = _uids(config) if not unfiltered else _unfiltered_uids(config)
    videos = [
        Video(
            uid=uid,
            path = _path_for(config, uid),
            frame_count = get_frame_count(_path_for(config, uid)),
            w = get_w(_path_for(config, uid)),
            h = get_h(_path_for(config, uid)),
            frame_rate = get_frame_rate(_path_for(config, uid)),
            has_audio = False,
            is_stereo = False,
        )
        for uid in uids
    ]
    if config.exclude_no_audio:
        return [v for v in videos if v.has_audio]

    return videos


# return the videos
def get_videos(config: FeatureExtractConfig) -> Tuple[List[Video], List[Video]]:
    """
    Return (videos_to_process, all_videos)
    """
    possibly_filtered_videos = _videos(config.io, unfiltered=False)
    all_videos = _videos(config.io, unfiltered=True)
    if config.io.video_limit > 0:
        random.shuffle(possibly_filtered_videos)
        return possibly_filtered_videos[0 : config.io.video_limit], all_videos
    return possibly_filtered_videos, all_videos
