import math

import torch
import torchvision
import numpy as np


class MixUpAug(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg.AUGS.MIXUP.ALPHA

    @torch.no_grad()
    def forward(self, x, y):
        if torch.rand(1) < self.cfg.AUGS.MIXUP.PROB:
            x, y = mixup_anom(x,y, self.alpha)
        return x, y


@torch.no_grad()
def select_index(xb, yb, uni=True):
    s = xb.shape[0]
    if uni: return torch.randperm(s).to(xb.device)
    else:
        raise NotImplementedError


def mixup_anom(xb, yb, alpha=1.0):
    index = select_index(xb, yb, uni=True)
    batch_size = torch.tensor([xb.shape[0]])

    ls = torch.ones(batch_size) * .5
    ls = ls.to(xb.device)

    ls = ls[:,None,None,None] # [BS,C,H,W]

    mx = ls * xb + (1 - ls) * xb[index]
    yb = (yb + yb[index]).clamp(0,1)
    return mx, yb


class MSR(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale_up = cfg.AUGS.MSR.scale_up
        self.scale_down = cfg.AUGS.MSR.scale_down
        self.num_batches = 4

    def get_size(self, size, seed):
        imgsz = size[0]
        gs = 32
        rng = np.random.RandomState(seed)
        sz = rng.uniform(low=imgsz * self.scale_down, high= imgsz * self.scale_up + gs, size=(2,)) //gs * gs
        sfy = sz[0] / size[0]  # scale factor
        sfx = sz[1] / size[1]  # scale factor
        new_size = [math.ceil(x * sf / gs) * gs for x,sf in zip(size, [sfy, sfx])]  # new shape (stretched to gs-multiple)
        return new_size

    @torch.no_grad()
    def forward(self, xb, yb, cb, ms_seed=None, off=None):
        if off is not None and cb.L.n_epoch > off:
            return xb, yb

        b,c,h,w = xb.shape
        if ms_seed is None:
            base = cb.L.n_epoch + cb.cfg.TRAIN.SEED
            ms_seed = base + cb.L.n_batch // self.num_batches
        nh,nw = self.get_size((h,w), ms_seed)

        xb = torch.nn.functional.interpolate(xb, (nh,nw), mode='bilinear')
        yb = torch.nn.functional.interpolate(yb, (nh,nw), mode='bilinear')

        return xb, yb
