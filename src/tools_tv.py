import torch

import shallow as sh


@sh.utils.common.return_first_call
def get_guide_layer(x):
    bs, ch, w,h = x.shape
    tt = torch.arange(w) / w
    tt = tt.view(1,1,-1,1).repeat(bs, 1, 1, h)
    return tt.to(x.device)


@sh.utils.common.run_once
def run_once(idx, foo, *args, **kwargs):
    return foo(*args, **kwargs)


def batch_quantile(b, q=.01):
    br = b.view(b.shape[0],-1)
    rq = torch.quantile(br, dim=1, q=1-q).view(-1,1,1,1)
    lq = torch.quantile(br, dim=1, q=  q).view(-1,1,1,1)
    return torch.max(torch.min(b, rq), lq)


def norm_2d(xb, mode='batch', mean=None, std=None):
    if mean is None:
        mean, std = xb.mean(), xb.std()
    else:
        mean = torch.tensor(mean).view(1,1,1,1).to(xb)
        std = torch.tensor(std).view(1,1,1,1).to(xb)

    if mode == 'batch':
        xb = (xb - mean) / std
    elif mode == 'channel':
        xb = (xb - xb.mean((0,2,3), keepdims=True)) / xb.std((0,2,3), keepdims=True)
    elif mode == 'example':
        xb = (xb - xb.mean((1,2,3), keepdims=True)) / xb.std((1,2,3), keepdims=True)
    elif mode == 'chexamp':
        xb = (xb - xb.mean((2,3), keepdims=True)) / xb.std((2,3), keepdims=True)
    elif mode == 'freq_norm':
        xb = (xb - xb.mean((0,1,3), keepdims=True)) / xb.std((0,1,3), keepdims=True)
    else:
        raise Exception
    return xb, mean, std
