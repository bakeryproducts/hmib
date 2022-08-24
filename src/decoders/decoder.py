from functools import partial

import timm
import torch
from torch import nn as nn
from torch.nn import functional as F

from decoders.segformer import SaneSegFormerHead


def oh_my_god(s): return eval(s)


def check_eval(k, v, evaler=oh_my_god):
    if k.startswith('e_'):
        k = k[2:]
        try: v = evaler(v)
        except Exception as e: print('Eval error', e)
    return k, v



class DecoderBasicBlock(nn.Module):
    def __init__(self, in_channels,
                 skip_channels,
                 out_channels,
                 prehook=None,
                 block=timm.models.byobnet.BasicBlock,
                 **kwargs):
        super().__init__()

        inchans = in_channels + skip_channels
        self.prehook = prehook if prehook is not None else torch.nn.Identity()
        self.block = block(inchans, out_channels, **kwargs)


    def forward(self, x, skip=None):
        x = self.prehook(x)
        if skip is not None:
            x = torch.hstack([x, skip])
        x = self.block(x)
        return x


class RevDecoderBasicBlock(nn.Module):
    def __init__(self, in_channels,
                 skip_channels,
                 out_channels,
                 prehook=None,
                 use_skip=True,
                 block=timm.models.byobnet.BasicBlock,
                 **kwargs):
        super().__init__()
        self.prehook = prehook if prehook is not None else torch.nn.Identity()
        self.block = block(in_channels, out_channels, **kwargs)

        skip_channels = skip_channels if use_skip else 0
        self.down = nn.Conv2d(skip_channels + out_channels, out_channels, 1,1)
        self.use_skip = use_skip

    def forward(self, x, skip=None):
        x = self.block(x)
        x = self.prehook(x)
        if skip is not None and self.use_skip:
            x = torch.hstack([x, skip])
        x = self.down(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        block_kwargs,
        all_kwargs,
        use_bottleneck=False,
        # Block=DecoderBasicBlock,
        last_scale=False,
        Block=RevDecoderBasicBlock,
    ):
        super().__init__()

        _ec = list(encoder_channels[:-1])
        skip_channels = ([0]*(len(decoder_channels) - len(_ec)) + _ec)[::-1] # [256, 128, 64, 32, 0]

        in_channels = [encoder_channels[-1]] + decoder_channels[:-1]
        out_channels = decoder_channels

        self.bottleneck = torch.nn.Sequential(
        ) if use_bottleneck else torch.nn.Identity()

        blocks = []
        # print(in_channels, skip_channels, out_channels)
        for in_ch, skip_ch, out_ch, bkw in zip(in_channels, skip_channels, out_channels, block_kwargs):
            # resolve duplcates:
            # print(in_ch, skip_ch, out_ch)
            filtered_decoder_kwargs = {}
            for k,v in all_kwargs.items():
                if k not in bkw:
                    filtered_decoder_kwargs[k] = v

            db = Block(in_ch, skip_ch, out_ch, **bkw, **filtered_decoder_kwargs)
            blocks.append(db)

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.last_scale = last_scale

    def forward(self, *features):
        features = features[::-1]
        x = features[0]
        skips = list(features[1:])
        padded_skips = skips + [None] * (len(self.blocks) - len(skips)) # always do exact N decoder blocks from cfg

        xx = [x,]
        x = self.bottleneck(x)
        for block, skip in zip(self.blocks, padded_skips):
            x = block(x, skip)
            xx.append(x)

        if self.last_scale:
            x = nn.functional.interpolate(xx[-1], scale_factor=(self.last_scale, self.last_scale))
            xx[-1] = x
        return xx


def create_decoder(enc, dec):
    enc_channels = [b['ch'] for b in enc['blocks']]

    blocks_kwargs = []
    dec_channels = []
    for i, block in enumerate(dec['blocks']):
        block = dict(block)
        dec_ch = block.pop('ch')
        dec_channels.append(dec_ch)
        block = dict(check_eval(k, v) for k, v in block.items())
        blocks_kwargs.append(block)

    dec_all_stages = dict(dec.get('all_stages', {}))

    decoder = Decoder(
        encoder_channels=enc_channels,
        decoder_channels=dec_channels,
        block_kwargs=blocks_kwargs,
        all_kwargs=dec_all_stages,
        **dec.get('base', {}),
    )
    return decoder


def create_segdec(enc, dec, embedding_dim):
    enc_channels = [b['ch'] for b in enc['blocks']]

    blocks_kwargs = []
    dec_channels = []
    for i, block in enumerate(dec['blocks']):
        block = dict(block)
        dec_ch = block.pop('ch')
        dec_channels.append(dec_ch)
        block = dict(check_eval(k, v) for k, v in block.items())
        blocks_kwargs.append(block)

    decoder = SaneSegFormerHead(enc_channels[::-1], embedding_dim, dropout=0, **dec.get('base', {}), )
    return decoder
