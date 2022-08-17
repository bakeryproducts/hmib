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


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice


def loss_seg(pred, target, loss, cfg, reduction='mean'):
    assert reduction == 'mean'

    sege_l = None
    segd_l = None
    reg_l = None
    cls_l = None
    ssl_l = None

    if cfg.MODEL.ARCH != 'ssl':
        # yb = target['yb']
        pr = pred['cls']
        gt = target['cls']
        cls_l = loss['cls'](pr, gt)

        pr = pred['yb']
        gt = target['yb']

        MULTILABEL = True
        if MULTILABEL:
            num_classes = 5
            multilabel = torch.zeros_like(gt).repeat(1, num_classes, 1, 1)
            for j,i in enumerate(target['cls']):
                multilabel[j,i] = gt[j]
            gt = multilabel

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

    else:
        # SSL MODE
        y = pred['yb'] # x_rec if ssl
        x = target['xb']
        mask = target['mask']
        patch_size = 4
        in_chans = 3
        mask = mask.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, y, reduction='none')
        # loss_recon = F.mse_loss(x_rec, x, reduction='none')
        # loss_recon = F.l2_loss(x, x_rec, reduction='none')
        # print(loss_recon.shape, mask.shape, x.shape, x_rec.shape)
        ssl_l = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / in_chans
        # ssl_l = loss_recon.mean()


    return dict(seg_dice=segd_l, seg_ce=sege_l, reg=reg_l, cls=cls_l, ssl=ssl_l)
