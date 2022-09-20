import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from morph import Erosion2d, Dilation2d


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


def get_border(yb):
    e = Erosion2d(1,1, 3).cuda()
    d = Dilation2d(1,1, 9).cuda()
    re = e(yb)
    rd = d(yb)
    r = (rd - re) > .5
    return r


def loss_seg(pred, target, loss, cfg, reduction='mean'):
    assert reduction == 'mean'

    sege_l = None
    segd_l = None
    reg_l = None
    cls_l = None

    # pr = pred['cls']
    # gt = target['cls']
    # cls_l = loss['cls'](pr, gt)

    pr = pred['yb']
    gt = target['yb']

    # b,5,h,w
    # clsb = target['cls']
    # print(clsb.shape, clsb[:10])
    # for i, c in enumerate(clsb):
    #     for j in range(5):
    #         if j != c:
    #             # print(f'zeroing {i}')
    #             pr[i,j] -= 5

    segd_l = loss['seg_dice'](pr, gt)
    sege_l = loss['seg_ce'](pr, gt)

    # pp = pr[:, 2].unsqueeze(1)
    # gg = gt[:, 2].unsqueeze(1)
    # border = get_border(gg)
    # seg_border = torch.nn.functional.binary_cross_entropy_with_logits(pp[border], gg[border])
    # alpha = .01
    # sege_l = (1-alpha) * sege_l + alpha * seg_border

    # print(sege_l.cpu().detach().item(), sh.utils.common.st(gt.float()), sh.utils.common.st(pr.float()))
    if cfg.FEATURES.USE_DS:
        ds = pred['ds']
        ds = ds[:-2]
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
