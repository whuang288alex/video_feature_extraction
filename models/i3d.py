# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
from config import BaseModelConfig, InferenceConfig
from models.common import FeedVideoInput, Mirror
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo

from .i3d_arch import build_i3d


@dataclass
class ModelConfig(BaseModelConfig):
    pretrained_dataset: str = "rgb_kinetics"

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
    print("Loading I3D")    # takes RGB / XY input
    model = build_i3d(pretrained=config.pretrained_dataset)

    assert model is not None

    if patch_final_layer:
        model.logits = Identity()

    # Set to GPU or CPU
    model = FeedVideoInput(model)
    model = model.eval()
    model = model.to(inference_config.device)
    return model


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    transforms = [
        Lambda(lambda x: x[:, ::config.dilation]),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(config.mean, config.std),
        ShortSideScale(size=config.side_size),
        Mirror(),
    ]
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(transforms),
            )
        ]
    )
