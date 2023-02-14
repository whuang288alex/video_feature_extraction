# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
from config import BaseModelConfig, InferenceConfig
from models.common import ThreeCrop, Mirror
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


@dataclass
class ModelConfig(BaseModelConfig):
    model_name: str = "omnivore_swinB"
    input_type: str = "video"
    side_size: int = 224
    crop_size: int = 224
    dilation: int = 2
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)


class WrapModel(Module):
    def __init__(self, model: Module, input_type: str):
        super().__init__()
        self.model = model
        self.input_type = input_type

    def forward(self, x) -> torch.Tensor:
        input = x['video']
        if input.ndim == 5:
            out = self.model(input, input_type=self.input_type)
        elif input.ndim == 6:
            # evaluate one crop at a time
            out = [self.model(input[:, i], input_type=self.input_type) \
                    for i in range(input.shape[1])]
            out = torch.stack(out, dim=-1).mean(dim=-1)
        else:
            raise ValueError('invalid input size')
        return out


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    
    if config.use_remote:
        print("Loading remote Omnivore")
        model = torch.hub.load("facebookresearch/omnivore", model=config.model_name)

    if patch_final_layer:
        model.heads.image = Identity()
        model.heads.video = Identity()
        model.heads.rgbd = Identity()

    # Set to GPU or CPU
    model = WrapModel(model, config.input_type)
    model = model.eval()
    model = model.to(inference_config.device)
    return model


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    assert config.input_type == "video"
    transforms = [
        Lambda(lambda x: x[:, ::config.dilation]),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(config.mean, config.std),
        ShortSideScale(size=config.side_size),
    ]
    
    if config.center_crop:
        transforms.append(CenterCropVideo(config.crop_size))
    elif config.three_crop:
        transforms.append(ThreeCrop(config.crop_size))
    
    if config.mirror:
        transforms.append(Mirror())
    
    return ApplyTransformToKey(
        key="video",
        transform=Compose(transforms),
    )
