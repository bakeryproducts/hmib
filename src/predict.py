import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import fire
import torch
import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm

from block_utils import paste_crop, batcher
from infer import Inferer
from mask_utils import rle_encode
from tiff import BatchedTiffReader, save_tiff
from mp import parallel_block_read

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def select_organ_from_predict(yb, organ):
    # yb BCHW
    # copy ORGANS from data, better not to source data.py
    ORGANS = {k: i for i, k in enumerate(['prostate', 'spleen', 'lung', 'largeintestine', 'kidney'])}
    if organ and organ != 'none':
        yb = yb[:, ORGANS[organ]].unsqueeze(1)
    else:
        yb, _ = torch.max(yb, dim=1, keepdim=True)
    return yb


def log(*m):
    # TODO normal logging?
    pass
    # print(m)


def infer_image(dataloader, inferer, scale, pad_size, image_size, organ=None, interpolation_mode='bilinear'):
    H, W = image_size

    # Infer batch by batch
    mask = np.zeros((1, H, W), dtype=float)
    for batch_blocks, batch_coords in dataloader:
        batch_blocks = torch.from_numpy(np.stack(batch_blocks))
        batch_blocks = batch_blocks.cuda().float()
        # BCHW
        # Infer batch
        log('LOAD', batch_blocks.shape, batch_coords[0])
        batch_blocks = torch.nn.functional.interpolate(batch_blocks, scale_factor=(scale, scale), mode=interpolation_mode)

        log('INFER', batch_blocks.shape)
        batch_masks = inferer(batch_blocks.float())  # bchw, logit

        log('PREDICT', batch_masks.shape)
        batch_masks = select_organ_from_predict(batch_masks, organ)
        batch_masks = torch.nn.functional.interpolate(batch_masks, scale_factor=(1./scale, 1./scale), mode=interpolation_mode)
        batch_masks.sigmoid_()

        log('ORGAN', batch_masks.shape,)
        batch_masks = batch_masks.cpu()

        log('FINAL', batch_masks.shape)

        # Copy block mask to the original
        for block_mask, block_cd in zip(batch_masks, batch_coords):
            #log(block_mask.shape, block_mask.max(), block_cd)
            paste_crop(mask, block_mask, block_cd, pad_size)

    log('MASK', mask.shape)
    return mask


def image_file_generator(images_dir, images_csv=None):
    EXT = "tiff"
    # Load from images_dir
    if images_csv is None:
        for image_file in images_dir.rglob(f"*.{EXT}"):
            yield image_file, None
    # Load from dataframe
    else:
        df = pd.read_csv(images_csv)
        for row in df.itertuples():
            image_file = images_dir / f"{row.id}.{EXT}"
            if not image_file.exists():
                print(f"Image {image_file} doesn't exist, skipping")
                continue
            yield image_file, row.organ


def main(
    model_file,
    images_dir,
    image_meter_scale,
    network_scale,
    organ,
    output_dir=None,
    output_csv=None,
    config_file=None,
    block_size=512,
    pad_ratio=0.25,
    batch_size=4,
    threshold=0.5,
    tta=False,
    device=None,
    tta_merge_mode="mean",
    images_csv=None,
):
    """
    This function will infer all images from images_csv if given
    or images_dir and save the results in output_csv file in the
    format of df with columns "id" and "rle".

    Params
    ------
        model_file: str
            Path to .pth model file
        images_dir: str
            Path to images directory
        output_dir: str
            Path to the directory where output masks will be stored
        output_csv: str, optional, default None
            Path to .csv file where to store inference result
            If None then no .csv result will be stored
        config_file: str, optional, default None
            Path to .yml config file
        block_size: int, default 512
            Inference block size
        network_scale: float, default 1024/3000
            Scale of the network was trained
        pad_ratio: float, default 0.25
            Ratio of padding during the inference
        batch_size: int, default 4
            Batch size
        tta: bool
            Apply 8 TTA or not
        device: int, default None
            Device for inference, should be 0, 1, ..  or None for cpu
        tta_merge_mode: str
            One of [mean, max]
        images_csv: str, default None
            Path to .csv file with images metadata
            If you pass this path only images from this df will be
            used for inference
    """
    # Check cmd args
    if device is not None:
        # wont init all gups
        #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
        device = 'cuda'
    else:
        device = 'cpu'

    experiment_dir = Path(model_file).parent.parent.parent
    images_dir = Path(images_dir)

    if config_file is None: config_file = experiment_dir / "src" / "configs" / "u.yaml"
    if output_dir is None: output_dir = Path(experiment_dir) / "predicts"
    output_dir.mkdir(exist_ok=True)

    # Create TiffReader initializer
    # TiffReader = partial(
    #     BatchedTiffReader,
    #     pad_ratio=pad_ratio,
    #     batch_size=batch_size,
    # )

    #Create inferer
    inferer = Inferer.create(
        model_file,
        config_file,
        experiment_dir,
        device=device,
        threshold=threshold,
        tta=tta,
        tta_merge_mode=tta_merge_mode,
    )

    result = []
    gen = image_file_generator(images_dir, images_csv)


    for image_file, _ in tqdm(gen):
        print('\n \t', image_file)
        # TODO: image_meter_scale should be image specific
        scale = image_meter_scale / network_scale
        scaled_block_size = int(round(block_size / scale))
        pad_size = int(scaled_block_size * pad_ratio)
        img_size = rio.open(image_file).shape # well, small price for func reader
        dst = output_dir / image_file.parent.stem
        dst.mkdir(exist_ok=True)
        log("SCALE", scale)

        # with TiffReader(image_file, scaled_block_size) as image_reader:
        _image_reader = parallel_block_read(image_file, scaled_block_size, pad_ratio, num_processes=8)
        image_reader = batcher(_image_reader, batch_size)
        mask = infer_image(image_reader, inferer, scale, pad_size, img_size, organ=organ)

        if output_csv:
            result.append({
                "image_filename": image_file.name,
                "rle": rle_encode((mask[0] > threshold).astype(np.uint8))
            })

        if output_dir:
            save_tiff(dst / image_file.name, mask * 255)

    if output_csv:
        result = pd.DataFrame(result)
        result.to_csv(output_csv, index=False)


if __name__ == "__main__":
    fire.Fire(main)
