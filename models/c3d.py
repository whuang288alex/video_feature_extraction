# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple
import os

import torch
from src.config import BaseModelConfig, InferenceConfig
from models.common import FeedVideoInput, ThreeCrop, Mirror
from torchvision.transforms import (
    Compose, Resize, CenterCrop, FiveCrop, TenCrop, Lambda
) 
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms._transforms_video import  CenterCropVideo, NormalizeVideo
from .c3d_arch import build_c3d, load_mean

@dataclass
class ModelConfig(BaseModelConfig):
    pretrained_dataset: str = "sports1m"

    # transformation config
    side_size: int = 288
    dilation: int = 1
    mean: Tuple[float] = (0.5, 0.5, 0.5)
    std: Tuple[float] = (0.5, 0.5, 0.5)


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    
    print("Loading C3D")    # takes RGB / XY input
    model = build_c3d(config.pretrained_dataset)
    assert model is not None
    
    # Set to GPU or CPU
    model = FeedVideoInput(model)
    model = model.eval()
    model = model.to(inference_config.device)
    return model


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    
    mean = load_mean(config.pretrained_dataset)
    assert mean is not None
    mean = mean.cpu()
    
    transforms = [
        Lambda(lambda x: x[:, ::config.dilation]),
        Lambda(lambda x: x / 255.0),
        # NormalizeVideo(config.mean, config.std),
        ShortSideScale(size=config.side_size),
    ]
    
    # handle crops
    if config.crop == 'center':
        transforms.append(CenterCrop(config.crop_size))
        transforms.append(Lambda(lambda crops: crops - mean))
        
    elif config.crop == 'five_crops':
        transforms.append(FiveCrop(config.crop_size))
        transforms.append(Lambda(lambda crops: torch.stack(crops)))
        transforms.append(Lambda(lambda crops: crops - mean))
        
    elif config.crop == 'ten_crops':
        transforms.append(TenCrop(config.crop_size))
        transforms.append(Lambda(lambda crops: torch.stack(crops)))
        transforms.append(Lambda(lambda crops: crops- mean))
    else:
        raise NotImplementedError()
    
    # handle mirror
    if config.mirror:
        transforms.append(Mirror())
        
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(transforms),
            )
        ]
    )
