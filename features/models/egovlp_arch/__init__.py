import os

import torch
import torch.nn as nn

from .video_transformer import SpaceTimeTransformer

EGOVLP_PATH = os.path.join(os.path.dirname(__file__), 'assets/egovlp.pth')


class EgoVLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = SpaceTimeTransformer()
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.proj = nn.Linear(768, 256)

        # load pre-trained weights
        ckpt = torch.load(
            EGOVLP_PATH, map_location=lambda storage, loc: storage
        )
        self.load_state_dict(ckpt)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        """
        x: [bs, f, c, h, w]
        """
        x0 = self.model(x)  # (bs, 768)
        x1 = self.norm(x0)  # (bs, 768)
        x2 = self.proj(x1)  # (bs, 256)
        
        x = torch.cat([x0, x1, x2], dim=-1)

        return x


def build_egovlp():
    return EgoVLP()