import os
import os.path as osp
from functools import partial

import albumentations.augmentations.geometric.functional as AGF
import fire
import numpy as np
import pandas as pd

from block_utils import paste_crop
from infer import Inferer
from mask_utils import rle_encode
from tiff import BatchedTiffReader


def infer_image(image_reader, inferer, rle=True, organ=None):
    H, W = image_reader.shape

    # Infer batch by batch
    mask = np.zeros((1, H, W)).astype(bool if rle else np.float32)
    for batch_blocks, batch_coords in image_reader:
        # Infer batch
        batch_masks = inferer(batch_blocks, organ=organ)

        # Copy block mask to the original
        for block_mask, block_cd in zip(batch_masks, batch_coords):
            block_mask = AGF.scale(block_mask, image_reader.inv_network_scale)
            block_mask = block_mask.transpose((2, 0, 1))
            paste_crop(mask, block_mask, block_cd, image_reader.scaled_pad_size)

    # Build the result
    return rle_encode(mask) if rle else mask


def image_file_generator(images_dir, images_csv=None):
    EXT = ".tiff"

    # Load from images_dir
    if images_csv is None:
        for fname in os.listdir(images_dir):
            name, ext = osp.splitext(fname)
            if ext != EXT:
                continue

            image_id = int(name)
            image_file = osp.join(images_dir, fname)
            organ = None
            yield image_file, image_id, organ

    # Load from dataframe
    else:
        df = pd.read_csv(images_csv)
        for row in df.itertuples():
            image_file = osp.join(images_dir, f"{row.id}{EXT}")
            if not osp.exists(image_file):
                print(f"Image {image_file} doesn't exist, skipping")
                continue

            yield image_file, row.id, row.organ


def main(
    experiment_dir,
    model_file,
    config_file,
    images_dir,
    output_csv,
    block_size=512,
    network_scale=1024/3000,
    pad_ratio=0.25,
    batch_size=4,
    threshold=0.5,
    tta=False,
    to_gpu=False,
    tta_merge_mode="mean",
    images_csv=None,
):
    """
    This function will infer all images from images_csv if given
    or images_dir and save the results in output_csv file in the
    format of df with columns "id" and "rle".

    Params
    ------
        experiment_dir: str
            Path to dir where should be ./src dir
        model_file: str
            Path to .pth model file
        config_file:
            Path to .yml config file
        images_dir: str
            Path to images directory
        output_csv: str
            Path to .csv file where to store inference result
        block_size: int, default 512
            Inference block size
        network_scale: float, default 1024/3000
            Scale of the network was trained
        pad_ratio: float, default 0.25
            Ratio of padding during the inference
        batch_size: int, default 4
            Batch size
        tta:
            bla

        to_gpu: str, default "cpu"
            Device for inference, should be "cpu" or "cuda"
        tta_merge_mode: str
            One of [mean, max]
        images_csv: str, default None
            Path to .csv file with images metadata
            If you pass this path only images from this df will be
            used for inference
    """
    assert isinstance(tta, bool) and isinstance(to_gpu, bool)

    # Create TiffReader initializer
    TiffReader = partial(
        BatchedTiffReader,
        block_size=block_size,
        network_scale=network_scale,
        pad_ratio=pad_ratio,
        batch_size=batch_size,
    )

    # Create inferer
    inferer = Inferer.create(
        model_file,
        config_file,
        experiment_dir,
        to_gpu=to_gpu,
        threshold=threshold,
        tta=tta,
        tta_merge_mode=tta_merge_mode,
    )

    result = []
    for image_file, image_id, organ in image_file_generator(images_dir, images_csv):
        image_reader = TiffReader(image_file)
        result.append({
            "id": image_id,
            "rle": infer_image(image_reader, inferer, rle=True, organ=organ),
        })
        image_reader.close()

    result = pd.DataFrame(result)
    result.to_csv(output_csv, index=False)


if __name__ == "__main__":
    fire.Fire(main)
