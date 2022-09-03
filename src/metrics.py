import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def calc_dice(pred, target, eps=1e-6, reduce='none'):
    with torch.no_grad():
        target = (target > .5).half()

        b, *_ = pred.shape
        # b,c,h,w = pred.shape
        x = pred.contiguous().view(b,-1)
        y = target.contiguous().view(b,-1)
        intersection = (x * y).sum(1)
        dice = ((2. * intersection + eps) / (x.sum(1) + y.sum(1) + eps))
        if reduce == 'mean': dice = dice.mean()
        return dice # [b, 1]


def calc_score(pred, target, reduce='none'):
    threshold = .5
    with torch.no_grad():
        pred = (pred > threshold).float()
        score = calc_dice(pred, target, reduce=reduce)
    return score


def _loss_dict(foo):
    def wrapper(*args, **kwargs):
        return {'seg':foo(*args, **kwargs)}
    return wrapper


def loss_seg(pred, target, loss, cfg, reduction='mean'):
    assert reduction == 'mean'

    sege_l = None
    segd_l = None
    reg_l = None
    cls_l = None

    pr = pred['cls']
    gt = target['cls']
    cls_l = loss['cls'](pr, gt)

    pr = pred['yb']
    gt = target['yb']

    segd_l = loss['seg_dice'](pr, gt)
    sege_l = loss['seg_ce'](pr, gt)

    # print(sege_l.cpu().detach().item(), sh.utils.common.st(gt.float()), sh.utils.common.st(pr.float()))
    if cfg.FEATURES.USE_DS:
        ds = pred['ds']
        _,_,h,w = gt.shape
        segd_li = 0
        sege_li = 0
        for i, hm in enumerate(ds):
            hm = torch.nn.functional.interpolate(hm, (h,w))
            segd_li += loss['seg_dice'](hm, gt)
            sege_li += loss['seg_ce'](hm, gt)

        segd_li = segd_li / len(ds)
        sege_li = sege_li / len(ds)

        alpha = .1
        segd_l = (1 - alpha) * segd_l + alpha * segd_li
        sege_l = (1 - alpha) * sege_l + alpha * sege_li

    return dict(seg_dice=segd_l, seg_ce=sege_l, reg=reg_l, cls=cls_l)
