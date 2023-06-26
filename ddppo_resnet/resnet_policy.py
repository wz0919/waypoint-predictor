#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from habitat_baselines.rl.ddppo.policy import resnet


class PNResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        baseplanes: int = 32,
        ngroups: int = 16,
        spatial_size: int = 128,
        make_backbone=getattr(resnet, 'resnet50'),
    ):
        super().__init__()

        self._n_input_depth = 1 # observation_space.spaces["depth"].shape[2]
        spatial_size = 256 // 2 # observation_space.spaces["depth"].shape[0]

        self.running_mean_and_var = nn.Sequential()

        input_channels = self._n_input_depth
        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        final_spatial = int(
            spatial_size * self.backbone.final_spatial_compress
        )
        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial ** 2))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, depth_observations):
        cnn_input = []

        if self._n_input_depth > 0:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x
