import timm
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from basemodels import ClassificationHead
from buildingblocks import Adapter
from decoders.upernet import FPN_fuse, PSPModule


def model_select(cfg):
    enc_kwargs = dict(cfg.MODEL.ENCODER)
    dec_kwargs = dict(cfg.MODEL.DECODER)
    seghead_kwargs = dict(cfg.MODEL.SEGHEAD)

    archs = dict(unet=EncoderDecoder, upernet=UperNet)

    # POP AGAIN!
    encoder_fact = hydra.utils.instantiate(enc_kwargs.pop('runner'))
    decoder_fact = hydra.utils.instantiate(dec_kwargs.pop('runner'))

    seg = partial(archs[cfg.MODEL.ARCH],
                  cfg=cfg,
                  encoder_fact=encoder_fact,
                  encoder_cfg=enc_kwargs,
                  decoder_fact=decoder_fact,
                  decoder_cfg=dec_kwargs,
                  seg_cfg=seghead_kwargs)
    return seg


class EncoderDecoder(nn.Module):
    def __init__(self, cfg, encoder_fact, encoder_cfg, decoder_fact, decoder_cfg, seg_cfg):
        super().__init__()
        self.encoder, encoder_cfg = encoder_fact(encoder_cfg) # will update cfg with stage channels
        self.decoder = decoder_fact(encoder_cfg, decoder_cfg)

        dec_out = decoder_cfg['blocks'][-1]['ch']
        # dec_out = self.decoder.embedding_dim
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


class UperNet(nn.Module):
    def __init__(self, cfg, encoder_fact, encoder_cfg, decoder_fact, decoder_cfg, seg_cfg):
        super().__init__()
        self.encoder, encoder_cfg = encoder_fact(encoder_cfg) # will update cfg with stage channels
        enc_channels = [b['ch'] for b in encoder_cfg['blocks']]

        fpn_out = enc_channels[0]
        self.PPN = PSPModule(enc_channels[-1], )
        self.FPN = FPN_fuse(enc_channels, fpn_out=fpn_out)

        self.seg_head = nn.Conv2d(fpn_out, **seg_cfg) # TODO : full head
        torch.nn.init.constant_(self.seg_head.bias, -4.59)


    def forward(self, batch):
        x = batch['xb']
        input_size = (x.size()[2], x.size()[3])

        features = self.encoder(x)
        features[-1] = self.PPN(features[-1])
        fpn = self.FPN(features)
        x = self.seg_head(fpn)
        #print(x.shape)

        masks = F.interpolate(x, size=input_size, mode='bilinear')
        return dict(yb=masks, cls=None, ds=None)
