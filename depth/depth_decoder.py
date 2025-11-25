# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from depth.layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        #self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    # def upsample_and_align(self, x, target):
    #     """上采样 x 并对齐到 target 形状"""
    #     x = nn.functional.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=True)
    #     return x

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            #x = [upsample(x)]
            x = [nn.functional.interpolate(  # 使用 interpolate 明确指定尺寸
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=True  # 对齐角落像素
            )]
            if self.use_skips and i > 0:
                #x += [input_features[i - 1]]
                #x = [upsample(x, input_features[i - 1]), input_features[i - 1]]
                # 获取跳跃连接的特征图，并裁剪到与上采样后的尺寸一致
                skip = input_features[i - 1]
                _, _, h, w = x[0].shape
                skip = nn.functional.interpolate(skip, size=(h, w), mode="bilinear", align_corners=True)
                x += [skip]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
    # def forward(self, input_features):
    #     self.outputs = {}
    #
    #     x = input_features[-1]  # 最底层特征
    #     for i in range(4, -1, -1):
    #         x = self.convs[("upconv", i, 0)](x)
    #         x = [nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)]  # 2倍上采样
    #
    #         if self.use_skips and i > 0:
    #             skip = input_features[i - 1]
    #             x[0] = self.upsample_and_align(x[0], skip)  # 确保尺寸匹配
    #             x.append(skip)
    #
    #         x = torch.cat(x, dim=1)  # 拼接通道维度
    #         x = self.convs[("upconv", i, 1)](x)
    #
    #         if i in self.scales:
    #             self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
