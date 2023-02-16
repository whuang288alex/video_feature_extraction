# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple

from config import BaseModelConfig, InferenceConfig
from models.common import FeedVideoInput, ThreeCrop, Mirror
from pytorchvideo.models.hub.vision_transformers import mvit_base_16, mvit_base_32x3
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


@dataclass
class ModelConfig(BaseModelConfig):
    pretrained_dataset: str = "k400"

    # transformation config
    side_size: int = 224
    crop_size: int = 224
    dilation: int = 1
    mean: Tuple[float] = (0.45, 0.45, 0.45)
    std: Tuple[float] = (0.225, 0.225, 0.225)


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    
    if config.use_remote:
        assert config.pretrained_dataset in ("k400")
        if config.pretrained_dataset == "k400":
            print("Loading remote K400 MViT")
            model = mvit_base_32x3(pretrained=True)
        
    assert model is not None

    if patch_final_layer:
        model.head = Identity()

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
    ]
    
    if config.crop == "center":
        transforms.append(CenterCropVideo(config.crop_size))
    elif config.crop == "three_crops":
        transforms.append(ThreeCrop(config.crop_size))
    else :
        raise ValueError('Need to either use center crop or three crop')

    if config.mirror:
        transforms.append(Mirror())
        
    # image-based dataset
    if config.pretrained_dataset == "imagenet":
        # NOTE untested due to MViT imagenet not not available on torch hub
        transforms += [Lambda(lambda x: x.squeeze_(2))]
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(transforms),
            )
        ]
    )