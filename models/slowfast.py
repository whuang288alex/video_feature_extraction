# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from src.config import BaseModelConfig, InferenceConfig
from models.common import FeedVideoInputList, Mirror
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo
from .slowfast_arch import build_slowfast


@dataclass
class ModelConfig(BaseModelConfig):
    model_path: Optional[str] = None
    hub_path: Optional[str] = "slowfast_r101"
    slowfast_alpha: int = 4

    # transformation config
    side_size: int = 288    # (512, 288) for 16:9 videos, (384, 288) for 4:3 videos
    dilation: int = 2       # slowfast takes every other frame from 64-frame clips
    mean: Tuple[float] = (0.45, 0.45, 0.45)
    std: Tuple[float] = (0.225, 0.225, 0.225)


class GetFv(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        fv_s = x.shape[1]
        return x.view(bs, fv_s)


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    if config.model_path is not None:
        raise AssertionError("not supported yet")
        model = None
    else:
        if config.use_remote:
            assert config.hub_path is not None
            print("Loading remote slowfast...")
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", config.hub_path, pretrained=True
            )
        else:
            assert config.hub_path is not None
            print("Loading local slowfast ...")
            model = build_slowfast(config.hub_path)

    assert model is not None

    # replace fixed-sized pooling with adaptive pooling to allow 
    # flexible input size (original implementation assumes 256x256 input)
    if patch_final_layer:
        model.blocks[5].pool = nn.ModuleList(
            [nn.AdaptiveAvgPool3d(1), nn.AdaptiveAvgPool3d(1)]
        )
        model.blocks[6] = GetFv()

    # Set to GPU or CPU
    model = FeedVideoInputList(model)
    model = model.eval()
    model = model.to(inference_config.device)
    return model


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, slow_fast_alpha):

        super().__init__()
        self.slow_fast_alpha = slow_fast_alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            -3,
            torch.linspace(
                0, frames.shape[-3] - 1, frames.shape[-3] // self.slow_fast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    
    transforms = [
        Lambda(lambda x: x[:, ::config.dilation]),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(config.mean, config.std),
        ShortSideScale(size=config.side_size),
    ]
    
    if config.mirror:
        transforms.append(Mirror())
        
    transforms.append(PackPathway(config.slowfast_alpha))
    
    return ApplyTransformToKey(
        key="video",
        transform=Compose(transforms),
    )
