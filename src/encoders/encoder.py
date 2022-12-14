from pathlib import Path
from functools import partial

import timm
import torch
from torch import nn as nn
from torch.nn import functional as F

from encoders.swin import SwinTransformer
from encoders.mixt import mit_b0, mit_b1, mit_b3, mit_b4
from encoders.coat import CoatFlat


def oh_my_god(s):
    return eval(s)


def check_eval(k, v, evaler=oh_my_god):
    if k.startswith('e_'):
        k = k[2:]
        try: v = evaler(v)
        except Exception as e: print('Eval error', e)
    return k, v


class SwinE(SwinTransformer):
    @property
    def feature_info(self):
        d = []
        for i in range(len(self.layers)):
            d.append(dict(num_chs=self.layers[i].blocks[0].attn.qkv.in_features))
        return d


def load_pretrained_swin(m):
    st = torch.load('input/weights/upernet_swin_tiny_patch4_window7_512x512.pth')
    stf = {}
    for k, v in st['state_dict'].items():
        if k == 'backbone.patch_embed.proj.weight':
            #print(k, v.shape)
            v = v.repeat(1,2,1,1)[:,:4]
        if 'backbone' in k:
            #break
            k = k[9:]
            stf[k] = v

    m.load_state_dict(stf, strict=False)
    return m


def load_pretrained_mixt(m, p):
    st = torch.load(p)
    stf = {}
    for k, v in st['state_dict'].items():
        if k == 'backbone.patch_embed1.proj.weight':
            v = v.repeat(1,2,1,1)[:,:4]
        if 'backbone' in k:
            k = k[9:]
            stf[k] = v

    m.load_state_dict(stf, strict=False)
    return m


def create_mixt(enc_cfg):
    name = enc_cfg.pop('model_name')
    names = dict(mit_b0=mit_b0, mit_b1=mit_b1, mit_b3=mit_b3, mit_b4=mit_b4)
    assert name in names, (name, names)
    enc = names[name](**enc_cfg)
    if 'pretrained' not in enc_cfg:
        pretrs = dict(
            mit_b3=Path('input/weights/segformer.b3.1024x1024.city.160k.pth'),
            mit_b4=Path('input/weights/segformer.b4.512x512.ade.160k.pth'),
        )
        pretr = pretrs[name]
        assert pretr.exists(), pretr
        load_pretrained_mixt(enc, pretr)

    blocks = [{'ch': i} for i in enc.embed_dims]
    enc_cfg['blocks'] = blocks
    return enc, enc_cfg


def create_swin(enc_cfg):
    enc_cfg.pop('model_name')
    # print(enc_cfg)
    enc = SwinE(**enc_cfg)
    load_pretrained_swin(enc)

    blocks = []
    for stage in enc.feature_info:
        blocks.append(dict(ch=stage['num_chs']))
    enc_cfg['blocks'] = blocks
    return enc, enc_cfg



def create_coat(enc_cfg):
    """
        coat_lite_mini
        coat_lite_small
        coat_lite_tiny
        coat_mini
        coat_tiny
        coatnet_0_rw_224
        coatnet_1_rw_224
        coatnet_bn_0_rw_224
        coatnet_nano_rw_224
        coatnet_rmlp_1_rw_224
        coatnet_rmlp_nano_rw_224
    """

    name = enc_cfg.pop('model_name')
    if 'pretrained' not in enc_cfg:
        enc_cfg["pretrained"] = True
    enc = timm.create_model(name,
                            return_interm_layers=False,
                            out_features=['x1_nocls', 'x2_nocls', 'x3_nocls', 'x4_nocls'],
                            **enc_cfg,
                            )
    enc.return_interm_layers = True # init in timm is broken

    if name == 'coat_lite_medium' and enc_cfg['pretrained']:
        p = 'input/weights/coat_lite_medium_384x384_f9129688.pth'
        st = torch.load(p)
        stf = {}
        for k, v in st['model'].items():
            if k == 'patch_embed1.proj.weight':
                v = v.repeat(1,2,1,1)[:,:4]
            stf[k] = v
        enc.load_state_dict(stf, strict=False)


    blocks = [{'ch': i} for i in enc.embed_dims]
    enc_cfg['blocks'] = blocks
    enc = CoatFlat(enc)
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
