import timm
import hydra
import torch
import torch.nn as nn
from functools import partial

from basemodels import ClassificationHead
from buildingblocks import Adapter
# from decoders.decoder import create_decoder
# from encoders.encoder import create_encoder


def model_select(cfg):
    enc_kwargs = dict(cfg.MODEL.ENCODER)
    dec_kwargs = dict(cfg.MODEL.DECODER)
    seghead_kwargs = dict(cfg.MODEL.SEGHEAD)

    archs = dict(unet=EncoderDecoder, )

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
