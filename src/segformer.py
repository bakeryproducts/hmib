# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn

from mmcv.cnn.bricks import ConvModule


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(b, -1, h,w)
        return x


class SaneSegFormerHead(torch.nn.Module):
    def __init__(self, feature_channels, embedding_dim, dropout=0, last_scale=1, **kwargs):
        super().__init__()
        # self.lin_layers = torch.nn.ModuleList([MLP(ic, embedding_dim) for ic in feature_channels]) # TODO: conv 1
        self.lin_layers = torch.nn.ModuleList([nn.Conv2d(ic, embedding_dim, 1, 1) for ic in feature_channels])
        self.last_scale = last_scale

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 3, out_channels=embedding_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)

        self.out_channels = feature_channels
        # self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, *features):
        features = features[::-1]
        H, W = features[len(self.lin_layers)].shape[2:]
        resize = partial(nn.functional.interpolate, size=(H,W), mode='bilinear',align_corners=False)

        cc = []
        for i in range(len(self.lin_layers)):
            ll = self.lin_layers[i]
            f = features[i]
            c = ll(f)

            # if i < len(self.lin_layers) - 1:
            c = resize(c)
            # print(i, c.shape)
            cc.append(c)

        _c = torch.cat(cc, dim=1)
        x = self.linear_fuse(_c)
        x = self.dropout(x)
        xx = [x]

        if self.last_scale:
            x = nn.functional.interpolate(x, scale_factor=(self.last_scale, self.last_scale))
            xx = [x]
        return xx


class SegFormerHead(torch.nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, in_channels, feature_strides, embedding_dim, num_classes, dropout=0, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        #x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        resize = torch.nn.functional.interpolate

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
