# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
import clip
from config import BaseModelConfig, InferenceConfig
from models.common import ThreeCrop, Mirror
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from .clip_arch import load

@dataclass
class ModelConfig(BaseModelConfig):
    model_name: str = "clip"
    input_type: str = "video"
    side_size: int = 224
    crop_size: int = 224
    dilation: int = 1
    clip_to_image: int = 8
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)

class WrapModel(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, x) -> torch.Tensor:
        input = x['video']
        if input.ndim == 5:
            out = self.model(input[:,:,0])
        elif input.ndim == 6:
            # input.shape: [1(bs), 6(mirror + 3crop), 3(color), 8(T), 224(w), 224(h)]
            # output.shape: [1(bs), 512(feature), 6(mirror + 3crop)] -> [1, 512, 2(mirror), 3(3crop)]
            out = torch.stack([self.model(input[:, i, :, 0]) for i in range(input.shape[1])], dim = -1).view(input.shape[0], -1, 2, 3)
        else:
            raise ValueError('invalid input size')
        return out

def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    assert config.hub_path is not None
    model = load(config.hub_path, device = InferenceConfig.device).encode_image
    model = WrapModel(model)
    model.eval()
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
    else :
        raise ValueError('Need to either use center crop or three crop')

    if config.mirror:
        transforms.append(Mirror())
    
    return ApplyTransformToKey(
        key="video",
        transform=Compose(transforms),
    )
