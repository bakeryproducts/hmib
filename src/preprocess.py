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


def df_hubmap2(src, csv, with_organs=False):
    ori_images_dir = src / "train_images"
    df = pd.read_csv(csv)
    print(f'\n\ttotal length: {len(df)}')

    for i, row in df.iterrows():
        name = row.id
        if with_organs:
            image_path = ori_images_dir / row.organ / f"{name}.tiff"
        else:
            image_path = ori_images_dir / f"{name}.tiff"
        image = load_tiff(str(image_path), mode="hwc")
        mask = rle_decode(row.rle, image.shape[:2])
        yield name, image, mask, row


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


    ori_train_csv = src / "trv2.csv"
    gen = df_hubmap2(src, ori_train_csv, ori_train_csv)
    # gen = folder_gen(src)

    # Preprocess images
    for name, image, mask, _ in tqdm(gen):
        # Resize
        image = cv2.resize(image, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert True, 'do 3/4 central crop on hpa data'

        # Save
        # if subcrops:
        #     for i, (y1, x1, y2, x2) in enumerate(crop_generator(size, crop_size, crop_step)):
        #         cv2.imwrite(str(output_images_dir / f"{name}_{i}.png"), image[y1: y2, x1: x2])
        #         cv2.imwrite(str(output_masks_dir / f"{name}_{i}.png"), mask[y1: y2, x1: x2])

        # else:
        cv2.imwrite(str(output_images_dir / f"{name}.png"), image)
        cv2.imwrite(str(output_masks_dir / f"{name}.png"), mask)


def do_folder_scales(src, dst='input/preprocessed', co=None, ki=None, sp=None, lu=None, pr=None):
    src = Path(src)
    dst = Path(dst)
    assert src.exists(), 'PREPROCESS DATA FIRST, scripts/prepare_hmib'

    output_dirname = f"f_{src.stem}_rle_v2"
    output_data_dir = dst / output_dirname

    csv = src / 'data.csv'
    gen = df_hubmap2(src, csv, with_organs=True)

    SCALES = {
        'prostate':pr,
        'spleen':sp,
        'lung':lu,
        'largeintestine':co,
        'kidney':ki
    }
    # Preprocess images
    for name, image, mask, row in tqdm(gen):
        # Resize
        organ_scale = SCALES[row.organ]
        if organ_scale == 'none':
            continue

        image = cv2.resize(image, (0,0), fx=1/organ_scale, fy=1/organ_scale, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (0,0), fx=1/organ_scale, fy=1/organ_scale, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fdst = output_data_dir / (row.organ + f'_{organ_scale}') / 'images'
        fdst.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fdst / f"{name}.png"), image)

        fdst = output_data_dir / (row.organ + f'_{organ_scale}') / 'masks'
        fdst.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fdst / f"{name}.png"), mask)


if __name__ == "__main__":
    fire.Fire(do_folder_scales)
