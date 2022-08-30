import os
import os.path as osp

import cv2
import fire
import pandas as pd

from tiff import load_tiff


def dice(true_mask, pred_mask, eps=1e-6):
    intersection = true_mask * pred_mask
    return (2 * intersection.sum() + eps) / (true_mask.sum() + pred_mask.sum() + eps)


def get_mask_file_pairs(true_masks_dir, pred_masks_dir):
    true_fnames = os.listdir(true_masks_dir)
    true_fnames = dict(map(lambda fname: (osp.splitext(fname)[0], fname), true_fnames))

    pred_fnames = os.listdir(pred_masks_dir)
    pred_fnames = dict(map(lambda fname: (osp.splitext(fname)[0], fname), pred_fnames))

    pair_names = set(true_fnames.keys()) & set(pred_fnames.keys())

    result = dict()
    for name in pair_names:
        result[name] = {
            "true": osp.join(true_masks_dir, true_fnames[name]),
            "pred": osp.join(pred_masks_dir, pred_fnames[name]),
        }

    return result


def load_mask(mask_file):
    ext = osp.splitext(mask_file)[1]
    if ext.lower() in {".tif", ".tiff"}:
        mask = load_tiff(mask_file, mode="hwc")
    else:
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    if len(mask.shape) == 3:
        mask = mask[..., 0]

    assert len(mask.shape) == 2

    # [0, 255] -> [0, 1]
    mask[mask > 1] = 1

    return mask


def main(
    true_masks_dir,
    pred_masks_dir,
    dices_csv=None,
):
    pairs = get_mask_file_pairs(true_masks_dir, pred_masks_dir)

    dices = []
    for filename, pair in pairs.items():
        true_mask = load_mask(pair["true"])
        pred_mask = load_mask(pair["pred"])
        if pred_mask.shape != true_mask.shape:
            th, tw = true_mask.shape[:2]
            pred_mask = cv2.resize(pred_mask, (tw, th), interpolation=cv2.INTER_CUBIC)

        dices.append({
            "filename": filename,
            "dice": dice(true_mask, pred_mask),
        })

    dices = pd.DataFrame(dices)

    if dices_csv is not None:
        dices.to_csv(dices_csv, index=False)

    mean_dice = dices.dice.mean()
    print(f"Mean dice score: {mean_dice:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
