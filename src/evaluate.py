import sys
import os
from pathlib import Path
import os.path as osp

import cv2
import fire
import numpy as np
import pandas as pd
from loguru import logger

from tiff import load_tiff


def init_logs(off=False):
    logger.remove()
    logger.add(sys.stdout, level="WARNING")
    if not off:
        logger.remove()
        logger.add(sys.stdout, level="INFO")


def binarize(m, thr):
    m = (m > thr).astype(int)
    return m


def dice(true_mask, pred_mask, eps=1e-6):
    intersection = true_mask * pred_mask
    return (2 * intersection.sum() + eps) / (true_mask.sum() + pred_mask.sum() + eps)


def get_mask_file_pairs(true_masks_dir, pred_masks_dir, recursive=True, ext='*.[tiff png]*', in_masks_only=False):
    true_masks_dir = Path(true_masks_dir)
    pred_masks_dir = Path(pred_masks_dir)

    true_fnames = list(true_masks_dir.rglob(ext))
    if in_masks_only:
        true_fnames = [f for f in true_fnames if 'masks' in str(f)]

    true_fnames = {f.stem: f for f in true_fnames}
    logger.info(f"Found {len(true_fnames)} true mask files")

    pred_fnames = list(pred_masks_dir.rglob(ext))
    pred_fnames = {f.stem:f for f in pred_fnames}
    logger.info(f"Found {len(pred_fnames)} pred mask files")

    pair_names = set(true_fnames.keys()) & set(pred_fnames.keys())

    result = dict()
    for name in pair_names:
        result[name] = {
            "true": str(true_fnames[name]),
            "pred": str(pred_fnames[name]),
        }
    assert result, (true_fnames, pred_fnames)

    logger.info(f"Loaded total {len(result)} mask pairs for dice calculation")

    return result


def load_mask(mask_file):
    ext = osp.splitext(mask_file)[1]
    if ext.lower() in {".tif", ".tiff"}:
        mask = load_tiff(mask_file, mode="hwc")
    else:
        mask = cv2.imread(mask_file)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # print(mask.shape, mask.max(), mask.dtype)

    if len(mask.shape) == 3:
        mask = mask[..., 0]

    assert len(mask.shape) == 2
    return mask



def main(
    pred_masks_dir,
    true_masks_dir,
    thr,
    thr_max=None,
    thr_total=20,
    csv=None,
    loff=False,
    in_masks_only=False,
):
    if thr_max is None:
        single_run(pred_masks_dir, true_masks_dir, thr, csv, loff, in_masks_only)
    else:
        thrs = np.linspace(thr, thr_max, thr_total)
        dices = []
        for thr in thrs:
            dice = single_run(pred_masks_dir, true_masks_dir, thr, csv, loff, in_masks_only)
            dices.append(dice)
        idx = np.argmax(dices)
        logger.warning(f'\t\nBest dice {dices[idx]} @ thr {thrs[idx]: .3f}')


def single_run(
    pred_masks_dir,
    true_masks_dir,
    thr,
    csv=None,
    loff=False,
    in_masks_only=False,
):
    init_logs(loff)
    pairs = get_mask_file_pairs(true_masks_dir, pred_masks_dir, in_masks_only=in_masks_only)

    dices = []
    for filename, pair in pairs.items():
        true_mask = load_mask(pair["true"])
        pred_mask = load_mask(pair["pred"])

        if pred_mask.shape != true_mask.shape:
            logger.info(f'ERROR: Sizes dont match! {filename}, {pred_mask.shape}, {true_mask.shape}')
            th, tw = true_mask.shape[:2]
            pred_mask = cv2.resize(pred_mask, (tw, th), interpolation=cv2.INTER_CUBIC)

        true_mask = binarize(true_mask, .5)
        pred_mask = binarize(pred_mask, thr)

        if False:
            h,w = true_mask.shape
            # print(h,w)
            cy = h//2
            cx = w//2
            s = 768 #* scale // 3
            true_mask = true_mask[cy - s//2: cy + s//2, cx-s//2:cx+s//2]
            pred_mask = pred_mask[cy - s//2: cy + s//2, cx-s//2:cx+s//2]

        dices.append({
            "filename": filename,
            "dice": dice(true_mask, pred_mask),
        })

    dices = pd.DataFrame(dices)

    if csv is not None:
        dices.to_csv(csv, index=False)

    mean_dice = dices.dice.mean()
    std_dice = dices.dice.std()
    logger.warning(f"\tMean dice score @ {thr: .3f} : {mean_dice:.4f} +- {std_dice:.4f}")
    return mean_dice


if __name__ == "__main__":
    fire.Fire(main)
