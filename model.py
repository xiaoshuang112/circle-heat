import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

INPUT_SIZE = 320
HEATMAP_SIZE = 80


class DepthwiseBlock(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
        )


class TrayCenterNet(nn.Module):
    feature_indices = (1, 3, 8, 12)
    feature_channels = (16, 24, 48, 576)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.lateral = nn.ModuleList([nn.Conv2d(channels, 64, 1) for channels in self.feature_channels])
        self.refine = nn.ModuleList([DepthwiseBlock(64) for _ in range(3)])
        self.heatmap_head = nn.Conv2d(64, 1, 1)
        self.localizable_head = nn.Linear(576, 1)
        self.occupancy_head = nn.Linear(576, 1)

    def forward(self, image):
        features = []
        for index, layer in enumerate(self.backbone):
            image = layer(image)
            if index in self.feature_indices:
                features.append(image)
        pyramid = self.lateral[-1](features[-1])
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=features[level].shape[-2:], mode="bilinear", align_corners=False)
            pyramid = self.refine[level](pyramid + self.lateral[level](features[level]))
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        return {
            "heatmap": self.heatmap_head(pyramid).squeeze(1),
            "localizable": self.localizable_head(pooled).squeeze(1),
            "occupancy": self.occupancy_head(pooled).squeeze(1),
        }
