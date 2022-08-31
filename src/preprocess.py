#!/usr/bin/python3

from pathlib import Path

import fire
import cv2
import pandas as pd
from tqdm import tqdm

from mask_utils import rle_decode
from tiff import load_tiff


def crop_generator(size, crop_size, crop_step):
    for h_start in range(0, size - crop_size + 1, crop_step):
        for w_start in range(0, size - crop_size + 1, crop_step):
            yield h_start, w_start, h_start + crop_size, w_start + crop_size


def check_sizes(crop_size, size, crop_step):
    assert crop_size > 0, f"crop_size <= 0: {crop_size}"
    assert crop_size <= size, (crop_size < size)
    assert (size - crop_size) % crop_step == 0, "crop_size and crop_step should cover full image"


def folder_gen(src):
    images = list((src / 'images').rglob('*.tiff'))
    print(f'\n\ttotal length: {len(images)}')
    for ifn in images:
        image = load_tiff(str(ifn))
        mfn = src / 'masks' / ifn.name
        masks = load_tiff(str(mfn))
        yield ifn.stem, image, masks


def df_hubmap2(src):
    ori_images_dir = src / "train_images"
    ori_train_csv = src / "trv2.csv"
    df = pd.read_csv(ori_train_csv)
    print(f'\n\ttotal length: {len(df)}')

    for i, row in df.iterrows():
        name = row.id
        image = load_tiff(str(ori_images_dir / f"{name}.tiff"), mode="hwc")
        mask = rle_decode(row.rle, image.shape[:2])
        yield name, image, mask


def do_cuts(src='input/hmib', dst='input/preprocessed', scale=3, crop_size=512, crop_step=256):
    #check_sizes(crop_size, size, crop_step)
    subcrops = False#size != crop_size
    src = Path(src)
    dst = Path(dst)

    output_dirname = f"rle_{scale}_v2"
    if subcrops:
        overlap = crop_size - crop_step
        output_dirname += f"_c{scale}_o{overlap}"

    # Output directories
    output_data_dir = dst / output_dirname
    output_images_dir = output_data_dir / "images"
    output_masks_dir = output_data_dir / "masks"
    output_images_dir.mkdir(exist_ok=False, parents=True)
    output_masks_dir.mkdir(exist_ok=False, parents=True)


    gen = df_hubmap2(src)
    # gen = folder_gen(src)

    # Preprocess images
    for name, image, mask in tqdm(gen):
        # Resize
        image = cv2.resize(image, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save
        # if subcrops:
        #     for i, (y1, x1, y2, x2) in enumerate(crop_generator(size, crop_size, crop_step)):
        #         cv2.imwrite(str(output_images_dir / f"{name}_{i}.png"), image[y1: y2, x1: x2])
        #         cv2.imwrite(str(output_masks_dir / f"{name}_{i}.png"), mask[y1: y2, x1: x2])

        # else:
        cv2.imwrite(str(output_images_dir / f"{name}.png"), image)
        cv2.imwrite(str(output_masks_dir / f"{name}.png"), mask)


if __name__ == "__main__":
    fire.Fire(do_cuts)
