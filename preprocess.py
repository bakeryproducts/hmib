import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.tiff import load_tiff


def rle_decode(rle_str, mask_shape, mask_dtype=np.uint8):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def create_if_not_exist(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)


def crop_generator(size, crop_size, crop_step):
    for h_start in range(0, size - crop_size + 1, crop_step):
        for w_start in range(0, size - crop_size + 1, crop_step):
            yield h_start, w_start, h_start + crop_size, w_start + crop_size


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--size", type=int, default=1024, help="Image size after resize")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--crop_step", type=int, default=256)
    parser.add_argument("--data_dir", type=str, default="./input")

    return parser.parse_args()


def check_sizes(args):
    if args.crop_size <= 0:
        raise ValueError(f"crop_size <= 0: {args.crop_size}")

    if args.crop_size > args.size:
        raise ValueError(f"crop_size > size")

    if (args.size - args.crop_size) % args.crop_step != 0:
        raise ValueError(f"crop_size and crop_step should cover full image")


def main(args):
    check_sizes(args)

    data_dir = args.data_dir
    subcrops = args.size != args.crop_size

    # Original data paths
    ori_data_dir = osp.join(data_dir, "hmib")
    ori_images_dir = osp.join(ori_data_dir, "train_images")
    ori_train_csv = osp.join(ori_data_dir, "train.csv")

    # Output directory name
    output_dirname = f"rle_{args.size}"
    if subcrops:
        overlap = args.crop_size - args.crop_step
        output_dirname += f"_c{args.crop_size}_o{overlap}"

    # Output directories
    output_data_dir = osp.join(args.data_dir, "preprocessed", output_dirname)
    output_images_dir = osp.join(output_data_dir, "images")
    output_masks_dir = osp.join(output_data_dir, "masks")

    create_if_not_exist(output_images_dir)
    create_if_not_exist(output_masks_dir)

    # Load train csv
    df = pd.read_csv(ori_train_csv)

    # Preprocess images
    for row in tqdm(df.itertuples(), total=len(df), desc="Preprocessing images"):
        # Load image and mask
        image = load_tiff(osp.join(ori_images_dir, f"{row.id}.tiff"))
        mask = rle_decode(row.rle, image.shape[:2])

        # Resize
        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_CUBIC)

        # Save
        if subcrops:
            for i, (y1, x1, y2, x2) in enumerate(crop_generator(args.size, args.crop_size, args.crop_step)):
                cv2.imwrite(osp.join(output_images_dir, f"{row.id}_{i}.png"), image[y1: y2, x1: x2])
                cv2.imwrite(osp.join(output_masks_dir, f"{row.id}_{i}.png"), mask[y1: y2, x1: x2])

        else:
            cv2.imwrite(osp.join(output_images_dir, f"{row.id}.png"), image)
            cv2.imwrite(osp.join(output_masks_dir, f"{row.id}.png"), mask)


if __name__ == "__main__":
    main(parse_args())
