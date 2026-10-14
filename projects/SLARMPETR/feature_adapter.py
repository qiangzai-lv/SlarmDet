import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SLARMFeatureAdapter(BaseModule):
    """Project each camera's patch grid to PETR's single feature level."""

    def __init__(self, in_channels, out_channels=256, downsample=2):
        super().__init__()
        if downsample not in (1, 2):
            raise ValueError('downsample must be 1 or 2')
        self.in_channels = in_channels
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.spatial = nn.Conv2d(out_channels, out_channels, 3,
                                 stride=downsample, padding=1)

    def forward(self, feature):
        b, v, c, h, w = feature.shape
        if c != self.in_channels:
            raise ValueError(f'Expected {self.in_channels} channels, got {c}')
        output = self.spatial(self.proj(feature.reshape(b * v, c, h, w)))
        return [output.reshape(b, v, *output.shape[1:])]
