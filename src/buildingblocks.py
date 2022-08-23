from functools import partial

import timm
import torch
from torch import nn as nn
from torch.nn import functional as F

from encoders.swin import SwinTransformer
from decoders.segformer import SaneSegFormerHead




class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, dropout=0.2):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True)
        linear = nn.Linear(in_channels, classes, bias=True)
        super().__init__(pool, flatten, dropout, linear)


def oh_my_god(s):
    return eval(s)


class SwinE(SwinTransformer):
    @property
    def feature_info(self):
        d = []
        for i in range(len(self.layers)):
            d.append(dict(num_chs=self.layers[i].blocks[0].attn.qkv.in_features))
        return d


def load_pretrained(m):
    st = torch.load('input/weights/upernet_swin_tiny_patch4_window7_512x512.pth')
    stf = {}
    for k, v in st['state_dict'].items():
        if 'backbone' in k:
            #break
            k = k[9:]
            stf[k] = v
    m.load_state_dict(stf)
    return m


def create_swin(enc_cfg):
    enc_cfg.pop('model_name')
    enc = SwinE(**enc_cfg)
    load_pretrained(enc)

    blocks = []
    for stage in enc.feature_info:
        blocks.append(dict(ch=stage['num_chs']))
    enc_cfg['blocks'] = blocks
    return enc, enc_cfg


def extend_timm(d):
    predef_timm = dict(pretrained=True, features_only=True)
    for k, v in predef_timm.items():
        if k not in d:
            d[k] = v
    return d


def create_encoder(enc_cfg):
    # enc = create_custom(enc_cfg)
    enc_cfg = extend_timm(enc_cfg)
    enc = create_timm(enc_cfg)

    blocks = []
    for stage in enc.feature_info:
        blocks.append(dict(ch=stage['num_chs']))
    enc_cfg['blocks'] = blocks
    return enc, enc_cfg


def create_timm(enc_cfg):
    enc_cfg = dict(enc_cfg)
    enc = timm.create_model(**enc_cfg)
    return enc


def create_custom(enc):
    in_channels = [enc['in_channels']]
    blocks = []
    for i, block in enumerate(enc['blocks']):
        block = dict(block)
        ch = block.pop('ch')
        in_channels.append(ch)
        blocks.append(block)

    # print(blocks)
    out_channels = [o for o in in_channels[1:]]
    in_channels.pop(-1)

    blocks_kwargs = []
    for i, block in enumerate(blocks):
        d = {}
        if 'down' not in block:
            down = torch.nn.MaxPool2d(2,2)
            d['pre_hook'] = down
        else:
            try:
                down = oh_my_god(block['down'])
            except Exception as e:
                print('READ CFG ERROR', e)
                down = torch.nn.MaxPool2d(2,2)
            d['pre_hook'] = down

        block.update(d)
        blocks_kwargs.append(block)
        # print(block)

    all_stages = dict(enc.get('all_stages', {}))
    Block = timm.models.byobnet.BasicBlock
    encoder = Encoder(in_channels,
                      out_channels,
                      Block,
                      blocks_kwargs,
                      all_stages,
                      **enc.get('base', {}),
                      )
    return encoder


def check_eval(k, v, evaler=oh_my_god):
    if k.startswith('e_'):
        k = k[2:]
        try: v = evaler(v)
        except Exception as e: print('Eval error', e)
    return k, v


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


def create_segdec(enc, dec):
    enc_channels = [b['ch'] for b in enc['blocks']]

    blocks_kwargs = []
    dec_channels = []
    for i, block in enumerate(dec['blocks']):
        block = dict(block)
        dec_ch = block.pop('ch')
        dec_channels.append(dec_ch)
        block = dict(check_eval(k, v) for k, v in block.items())
        blocks_kwargs.append(block)

    embedding_dim = 128
    decoder = SaneSegFormerHead(enc_channels[::-1], embedding_dim, dropout=0, **dec.get('base', {}), )
    return decoder


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, Block, block_kwargs, kwargs):
        super().__init__()

        blocks = []
        for ic, oc, bkw in zip(in_channels, out_channels, block_kwargs):
            # resolve duplcates:
            filtered_decoder_kwargs = {}
            for k,v in kwargs.items():
                if k not in bkw:
                    filtered_decoder_kwargs[k] = v

            db = Block(ic, oc, **bkw, **filtered_decoder_kwargs)
            blocks.append(db)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


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


class Adapter(torch.nn.Module):
    def __init__(self, decoder_channels, out_channel):
        super().__init__()
        head_dim = out_channel
        adapter_channels = decoder_channels[:-1]
        self.convs = nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(d, head_dim, (1,1)),
                torch.nn.BatchNorm2d(head_dim),
                torch.nn.ReLU(),
            )
            for d in adapter_channels])

    def forward(self, features):
        features = features[1:-1] # drop decoder input and full size
        rr = []
        assert len(features) == len(self.convs), (len(features),len(self.convs))
        for i, f in enumerate(features):
            r = self.convs[i](f)
            rr.append(r)
        return rr
