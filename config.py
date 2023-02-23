# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import importlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from torchvision.transforms import Compose, Lambda

import cv2


@dataclass
class NormalizationConfig:
    normalize_audio: bool = False
    resample_audio_rate: int = 16000
    resampling_method: str = "sinc_interpolation"


@dataclass
class InputOutputConfig:
    """
    Assumptions:
    1. videos are in a single subdirectory with the pattern: <dir>/<uid>.mp4
    2. the manifest file exists in the same subdirectory

    If you need to adjust any of these assumptions, then please refer to:
    - `_path_for`, and
    - `_uid_to_frame_count`
    """

    # input
    filter_completed: bool = True
    video_dir_path: str = "/datasets01/ego4d_track2/v1/full_scale/"
    ego4d_download_dir: str = "/checkpoint/miguelmartin/ego4d/"
    uid_list: Optional[List[str]] = None
    video_limit: int = -1
    debug_mode: bool = False
    debug_path: str = "/checkpoint/miguelmartin/ego4d_track2/v1/debug_frames"

    # output
    out_path: str = (
        "/checkpoint/miguelmartin/ego4d_track2_features/full_scale/v1_1/action_features"
    )
    exclude_no_audio: bool = False


@dataclass
class InferenceConfig:
    device: str = "cuda"

    # 0 == don't use dataloader
    # >0 use dataloader with bs=batch_size
    batch_size: int = 1

    # only used if batch_size != 0
    num_workers: int = 9

    prefetch_factor: int = 2

    fps: int = 30
    frame_window: int = 32
    stride: int = 16
    include_audio: bool = False
    include_video: bool = True
    norm_config: NormalizationConfig = NormalizationConfig()


@dataclass
class ScheduleConfig:
    run_locally: bool = False

    log_folder: str = "slurm_log/%j"

    # Scheduler Configuration
    timeout_min: int = int(12 * 60)
    constraint: str = "volta"
    slurm_partition: str = "pixar"
    slurm_array_parallelism: int = 256
    gpus_per_node: int = 1
    cpus_per_task: int = 10

    # Batching Configuration
    overhead: float = 2  # off in the worst case -- estimate will be wrong
    time_per_forward_pass: float = 0.8
    schedule_time_per_node: float = 10


@dataclass
class BaseModelConfig:
    center_crop = False
    three_crop = False
    mirror = False


@dataclass
class ModelConfig(BaseModelConfig):
    model_path: Optional[str] = None
    hub_path: Optional[str] = "slowfast_r101"
    slowfast_alpha: int = 4
    side_size: int = 288   
    dilation: int = 2       
    mean: Tuple[float] = (0.45, 0.45, 0.45)
    std: Tuple[float] = (0.225, 0.225, 0.225)
    
    
@dataclass(order=True)
class FeatureExtractConfig:
    io: InputOutputConfig
    inference_config: InferenceConfig
    schedule_config: ScheduleConfig
    model_config: BaseModelConfig
    model_module_str: str = ""
    force_yes: bool = False

    
@dataclass
class Video:
    """
    Description of a video
    """
    uid: str
    path: str
    frame_count: int
    w: int
    h: int
    has_audio: bool
    is_stereo: bool = False
    frame_rate: int = 30

    @property
    def dim(self) -> int:
        return (self.w * self.h) / (2 if self.is_stereo else 1)


# helper function to get model specific functions
def get_model_module(config: FeatureExtractConfig):
        return importlib.import_module(config.model_module_str)


def get_transform(config: FeatureExtractConfig) -> Any:
    ic = config.inference_config
    nc = ic.norm_config
    model_transform = get_model_module(config).get_transform(ic, config.model_config)
    transforms = []
    if hasattr(config, "norm_config") and config.norm_config.normalize_audio:
        print(f"Normalizing with: {config.norm_config}")

        def resample_audio(x):
            return Resample(
                orig_freq=x["audio_sample_rate"],
                new_freq=nc.resample_audio_rate,
                resampling_method=nc.resampling_method,
            )

        transforms += [Lambda(resample_audio)]

    transforms += [model_transform]
    return Compose(transforms)


def load_model(config: FeatureExtractConfig, patch_final_layer: bool = True) -> Any:
    module = get_model_module(config)
    return module.load_model(
        config.inference_config,
        config.model_config,
        patch_final_layer=patch_final_layer,
    )


# helper function to get all the videos in the input directory
def _videos(config: InputOutputConfig, unfiltered: bool = False) -> List[Video]:
    
    def _uids_for_dir(path: str) -> List[str]:
        ret = [
            p
            for p in os.listdir(path)
            if Path(p).suffix not in [".json", ".csv", ".csv"]
            and not p.startswith(".")
            and not p.startswith("manifest")
        ]
        return [Path(p).stem for p in ret]
    
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

    def _path_for(config: InputOutputConfig, uid: str) -> str:
        return f"{config.video_dir_path}/{uid}.mp4"

    def get_frame_count(video_path):
        vid = cv2.VideoCapture(video_path)
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        return length

    def get_frame_rate(video_path):
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        # print(fps)
        return int(fps)
    
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
            has_audio = False,
            is_stereo = False,
            frame_rate = get_frame_rate(_path_for(config, uid))
        )
        for uid in uids
    ]
    if config.exclude_no_audio:
        return [v for v in videos if v.has_audio]

    return videos


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


@hydra.main(config_path="configs", config_name=None)
def test_load_config(config: FeatureExtractConfig):
    print(
            f"""
                Config:
                
                {OmegaConf.to_yaml(config)}
             """
    )


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(
        name="default",
        node=FeatureExtractConfig(
            io=InputOutputConfig(),
            inference_config=InferenceConfig(),
            schedule_config=ScheduleConfig(),
            model_config=ModelConfig(),
        ),
    )
    test_load_config()  # pyre-ignore
