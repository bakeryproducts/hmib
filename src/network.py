import timm
import torch
import torch.nn as nn
from functools import partial

from basemodels import ClassificationHead
from buildingblocks import create_encoder, create_decoder, create_swin, Adapter, create_segdec


def model_select(cfg):
    enc_kwargs = dict(cfg.MODEL.ENCODER)
    dec_kwargs = dict(cfg.MODEL.DECODER)
    seghead_kwargs = dict(cfg.MODEL.SEGHEAD)

    archs = dict(unet=EncoderDecoder, )

    seg = partial(archs[cfg.MODEL.ARCH], cfg=cfg, encoder_cfg=enc_kwargs, decoder_cfg=dec_kwargs, seg_cfg=seghead_kwargs)
    return seg


class EncoderDecoder(nn.Module):
    def __init__(self, cfg, encoder_cfg, decoder_cfg, seg_cfg):
        super().__init__()
        self.encoder, encoder_cfg = create_encoder(encoder_cfg) # will update cfg with stage channels
        # self.encoder, encoder_cfg = create_swin(encoder_cfg) # will update cfg with stage channels
        # self.decoder = create_decoder(encoder_cfg, decoder_cfg)
        self.decoder = create_segdec(encoder_cfg, decoder_cfg)

        dec_out = decoder_cfg['blocks'][-1]['ch']
        self.seg_head = nn.Conv2d(dec_out, **seg_cfg) # TODO : full head
        torch.nn.init.constant_(self.seg_head.bias, -4.59)

        num_classes = 5
        self.cls_head = ClassificationHead(encoder_cfg['blocks'][-1]['ch'], num_classes)
        self.ds_adapter = Adapter(self.decoder.out_channels, 1) if cfg.FEATURES.USE_DS else torch.nn.Identity()


    def forward(self, batch):
        x = batch['xb']

        features = self.encoder(x)
        cls = self.cls_head(features[-1])

        xx = self.decoder(*features)
        # print([x.shape for x in xx])
        last_feat = xx[-1]
        masks = self.seg_head(last_feat)
        deep_supervision = self.ds_adapter(xx)

        return dict(yb=masks, cls=cls, ds=deep_supervision)
