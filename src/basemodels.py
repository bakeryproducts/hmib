import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, dropout=0.2):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True)
        linear = nn.Linear(in_channels, classes, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class SegmentationModel(torch.nn.Module):
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        labels1 = self.classification_head1(features[-1])
        labels2 = self.classification_head2(features[-1])
        return dict(yb=masks, posb=labels1, keyf=labels2)


class DecoderBasicBlock(nn.Module):
    def __init__(
            self,
            inplanes,
            skip_channels,
            planes,
            attn=None,
            prehook=None,
            base_block=timm.models.resnet.BasicBlock,
            **kwargs):
        super().__init__()

        inchans = inplanes + skip_channels
        self.block = base_block(inchans, planes, **kwargs)
        self.prehook = prehook or torch.nn.Identity()
        self.attn = attn or torch.nn.Identity()

    def forward(self, x, skip=None, upscale=True):
        x = self.prehook(x)
        if skip is not None:
            x = torch.hstack([x, skip])
        x = self.block(x)
        x = self.attn(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            DecoderBlock,
            blocks_kwargs,
            decoder_kwargs,
            encoder_head=False,
            bottleneck=False,
            debug=False):
        super().__init__()
        self.debug = debug
        skip_channels = ([0] + list(encoder_channels[:-1]))[::-1]
        in_channels = [encoder_channels[-1]] + decoder_channels[:-1]
        out_channels = decoder_channels

        self.bottleneck = bottleneck or torch.nn.Identity()

        blocks = []
        for in_ch, skip_ch, out_ch, bkw in zip(in_channels, skip_channels, out_channels, blocks_kwargs):
            # resolve duplcates:
            filtered_decoder_kwargs = {}
            for k,v in decoder_kwargs.items():
                if k not in bkw:
                    filtered_decoder_kwargs[k] = v

            block = DecoderBlock(in_ch, skip_ch, out_ch, **bkw, **filtered_decoder_kwargs)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def log(self, *m):
        if self.debug: print(m)

    def forward(self, *features):
        features = features[::-1]
        x = features[0]
        skips = list(features[1:]) + [None]

        xx = [x,]
        x = self.bottleneck(x)
        for db, skip in zip(self.blocks, skips):
            self.log(x.shape, db)
            x = db(x, skip)
            xx.append(x)

        return x


def get_head(in_features, out_channels, scales, drop_block_rate=.0, use_aa=False, attention_type=None):
    # assert isinstance(scales, list), scales
    out_chs = [in_features // sc for sc in scales]

    blocks = []
    in_ch = in_features
    for ch in out_chs:
        block = get_bb(in_ch, ch, use_aa, attention_type, drop_block_rate)
        in_ch = ch
        blocks.append(block)

        # after first block, dont use att, aa, drop reg
        attention_type = None
        drop_block_rate = 0.

    seg_head = torch.nn.Sequential(
        *blocks,
        smp.base.SegmentationHead(out_chs[-1], out_channels))
    return seg_head


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0, upsampling=1):
        dropout = nn.Dropout2d(dropout)
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(dropout, conv2d, upsampling)


def get_simple_head(in_features, out_channels, scales, dropout=.0, upsampling=1):
    seg_head = SegmentationHead(in_features, out_channels, dropout=dropout, upsampling=upsampling)
    return seg_head


def get_staged_head(in_features, out_channels, scales, drop_block_rate=.0, use_aa=False, attention_type=None):
    # assert isinstance(scales, list), (scales, type(scales))
    out_chs = [in_features // sc for sc in scales]
    stage = get_stage(in_features, out_features=out_chs[-1], depth=len(out_chs), drop_block_rate=drop_block_rate)
    seg_head = torch.nn.Sequential(stage, smp.base.SegmentationHead(out_chs[-1], out_channels))
    return seg_head


def get_stage(in_features, out_features, depth, drop_block_rate):
    dp_rates = [drop_block_rate] * depth
    s = timm.models.convnext.ConvNeXtStage(in_features,
                                           out_features,
                                           stride=1,
                                           depth=depth,
                                           dp_rates=dp_rates,
                                           norm_layer=timm.models.convnext.LayerNorm2d)
    return s


def get_bb(in_features, out_features, use_aa, attention_type, drop_block_rate):
    # attn_layer = timm.models.layers.get_attn('gc') if use_att else None
    attn_layer = timm.models.layers.get_attn(attention_type) if attention_type is not None else None
    aa_layer = timm.models.resnet.BlurPool2d if use_aa else None
    drop_layer = timm.models.layers.drop.DropBlock2d(drop_block_rate, 3, 1.0) if drop_block_rate else None
    downsample = timm.models.resnet.downsample_conv(in_features, out_features, kernel_size=3, stride=1)
    return timm.models.resnet.BasicBlock(in_features,
                                         planes=out_features,
                                         stride=1,
                                         dilation=1,
                                         drop_block=drop_layer,
                                         downsample=downsample,
                                         attn_layer=attn_layer,
                                         aa_layer=aa_layer)
