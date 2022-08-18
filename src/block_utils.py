import numpy as np


def generate_block_coords(H, W, block_size):
    h,w = block_size
    nYBlocks = (int)((H + h - 1) / h)
    nXBlocks = (int)((W + w - 1) / w)

    for X in range(nXBlocks):
        cx = X * h
        for Y in range(nYBlocks):
            cy = Y * w
            yield cy, cx, h, w


def pad_block(y, x, h, w, pad):
    return np.array([y - pad, x - pad, h + 2 * pad, w + 2 * pad])


def crop(src, y, x, h, w):
    return src[..., y: y + h, x: x + w]


def paste(src, block, y, x, h, w):
    src[..., y: y + h, x: x + w] = block


def paste_crop(src, part, block_cd, pad):
    H, W = src.shape[-2:]
    y, x, h, w = block_cd
    h, w = min(h, H - y), min(w, W - x)
    part = crop(part, pad, pad, h, w)
    paste(src, part, *block_cd)
