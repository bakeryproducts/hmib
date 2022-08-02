import timm
import torch
import torch.nn as nn
from functools import partial

from basemodels import SegmentationModel, DecoderBasicBlock, UnetDecoder, get_staged_head, get_head, get_simple_head
from buildingblocks import create_encoder, create_decoder, create_swin

from swin import SwinTransformerForSimMIM
from vit import VisionTransformerForSimMIM


def model_select(cfg):
    enc_kwargs = dict(cfg.MODEL.ENCODER)
    dec_kwargs = dict(cfg.MODEL.DECODER)
    seghead_kwargs = dict(cfg.MODEL.SEGHEAD)

    archs = dict(unet=SaneUnet, ssl=SSL)

    seg = partial(archs[cfg.MODEL.ARCH], encoder_cfg=enc_kwargs, decoder_cfg=dec_kwargs, seg_cfg=seghead_kwargs)
    return seg


class SimMIM(nn.Module):
    def __init__(self, in_chans, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                # in_channels=self.encoder.num_features[-1],
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        features = self.encoder(x, mask)
        x_rec = self.decoder(features[-1])
        return x_rec


class SSL(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, seg_cfg, stride=32):
        super().__init__()
        encoder_cfg.pop('model_name')
        # self.s = SwinTransformerForSimMIM(**encoder_cfg)
        self.s = SwinTransformerForSimMIM(img_size=192,
                                          win_size=6,
                                          embed_dim=128,
                                          depths=[2, 2, 18, 2],
                                          num_heads=[4,8,16,32],
                                          **encoder_cfg)
        # self.s = VisionTransformerForSimMIM(norm_layer=partial(nn.LayerNorm, eps=1e-6), **encoder_cfg)
        self.sm = SimMIM(3, self.s, stride)
        # self.p = torch.nn.Parameter(torch.ones(1))

    def forward(self, batch):
        x = batch['xb']
        mask = batch['mask']
        # print(mask.shape, x.shape)

        r = self.sm(x, mask) # B,C,H,W
        # r = self.p * x

        cls = torch.zeros(1).cuda()
        return dict(yb=r, cls=cls)


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, dropout=0.2):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True)
        linear = nn.Linear(in_channels, classes, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class SaneUnet(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, seg_cfg):
        super().__init__()
        self.encoder, encoder_cfg = create_encoder(encoder_cfg) # will update cfg with stage channels
        # self.encoder, encoder_cfg = create_swin(encoder_cfg) # will update cfg with stage channels
        self.decoder = create_decoder(encoder_cfg, decoder_cfg)

        dec_out = decoder_cfg['blocks'][-1]['ch']
        self.seg_head = nn.Conv2d(dec_out, **seg_cfg) # TODO : full head
        torch.nn.init.constant_(self.seg_head.bias, -4.59)

        num_classes = 5
        self.cls_head = ClassificationHead(encoder_cfg['blocks'][-1]['ch'], num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(dec_out)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=.2, inplace=True)
        self.linear = nn.Linear(encoder_cfg['blocks'][-1]['ch'], num_classes, bias=False)

    def forward(self, batch):
        x = batch['xb']

        features = self.encoder(x)
        cls = self.cls_head(features[-1])

        xx = self.decoder(*features)
        last_feat = xx[-1]
        masks = self.seg_head(last_feat)

        return dict(yb=masks, cls=cls)


def get_att_layer(d):
    try:
        attention_type = d.pop('attention_type')
        attn_layer = timm.models.layers.get_attn(attention_type)
    except KeyError:
        attn_layer = None
    return attn_layer
