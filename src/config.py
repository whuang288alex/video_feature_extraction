# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import importlib
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from torchvision.transforms import Compose, Lambda

import warnings
warnings.filterwarnings("ignore")

@dataclass
class NormalizationConfig:
    normalize_audio: bool = False
    resample_audio_rate: int = 16000
    resampling_method: str = "sinc_interpolation"


@dataclass
class InputOutputConfig:

    # input
    filter_completed: bool = True
    video_dir_path: str = "/inputs/videos_ego4d"
    ego4d_download_dir: str =  "/inputs"
    uid_list: Optional[List[str]] = None
    video_limit: int = -1
    debug_mode: bool = False
    debug_path: str = "/debug"

    # output
    out_path: str = "/features/c3d"
    exclude_no_audio: bool = False


@dataclass
class InferenceConfig:
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 8
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
    overhead: float = 2 
    time_per_forward_pass: float = 0.8
    schedule_time_per_node: float = 10


@dataclass
class BaseModelConfig:
    mirror = False
    crop: Optional[str] = None
    crop_size: int = 112 
    side_size: int = 288  
    dilation: int = 2 
    mean: Tuple[float] = (0.45, 0.45, 0.45)
    std: Tuple[float] = (0.225, 0.225, 0.225)


@dataclass
class ModelConfig(BaseModelConfig):   
    model_path: Optional[str] = None
    pretrained_dataset: Optional[str] = None
    hub_path: Optional[str] = None
    slowfast_alpha: int = 4
   
    
@dataclass(order=True)
class FeatureExtractConfig:
    io: InputOutputConfig
    inference_config: InferenceConfig
    schedule_config: ScheduleConfig
    model_config: BaseModelConfig
    model_module_str: str = ""
    
@dataclass
class Video:
    uid: str
    path: str
    w: int
    h: int
    frame_count: int
    frame_rate: int = 30
    has_audio: bool = False
    is_stereo: bool = False
    
    @property
    def dim(self) -> int:
        return (self.w * self.h) / (2 if self.is_stereo else 1)


# helper function to get model specific functions
def get_model_module(config: FeatureExtractConfig):
    return importlib.import_module(config.model_module_str)


# helper function to get model specific transform
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


# helper function to get model specific model
def load_model(config: FeatureExtractConfig, patch_final_layer: bool = True) -> Any:
    module = get_model_module(config)
    return module.load_model(
        config.inference_config,
        config.model_config,
        patch_final_layer=patch_final_layer,
    )


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
    test_load_config() 
