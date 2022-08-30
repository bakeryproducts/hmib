import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from block_utils import paste_crop
from infer import Inferer
from mask_utils import rle_encode
from tiff import BatchedTiffReader, save_tiff


def select_organ_from_predict(yb, organ=None):
    # yb BCHW
    # copy ORGANS from data, better not to source data.py
    ORGANS = {k: i for i, k in enumerate(['prostate', 'spleen', 'lung', 'largeintestine', 'kidney'])}
    if organ is not None:
        yb = yb[:, ORGANS[organ]].unsqueeze(1)
    else:
        yb, _ = torch.max(yb, dim=1, keepdim=True)
    return yb


def log(*m):
    pass
    # print(m)


def infer_image(image_reader, inferer, scale, rle_threshold=0.5, organ=None):
    dataloader = image_reader
    # dataloader = DataLoader(image_reader, num_workers=2, batch_size=image_reader.batch_size)

    H, W = image_reader.shape

    # Infer batch by batch
    mask = np.zeros((1, H, W), dtype=float)
    for batch_blocks, batch_coords in tqdm(dataloader):
        # BCHW
        # Infer batch
        log('LOAD', batch_blocks.shape, batch_coords[0])
        batch_blocks = torch.nn.functional.interpolate(
            batch_blocks, scale_factor=(scale, scale))

        log('INFER', batch_blocks.shape, batch_blocks.max())
        batch_masks = inferer(batch_blocks)  # bchw, logit

        log('PREDICT', batch_masks.shape, batch_masks.max())
        batch_masks = select_organ_from_predict(batch_masks, organ)
        batch_masks = torch.nn.functional.interpolate(
            batch_masks, scale_factor=(1./scale, 1./scale))
        batch_masks.sigmoid_()

        log('ORGAN', batch_masks.shape,)
        batch_masks = batch_masks.cpu()

        log('FINAL', batch_masks.shape)

        # Copy block mask to the original
        for block_mask, block_cd in zip(batch_masks, batch_coords):
            log(block_mask.shape, block_mask.max(), block_cd)
            paste_crop(mask, block_mask, block_cd, image_reader.pad_size)

    log('MASK', mask.shape, mask.mean(), mask.max())

    rle = rle_encode((mask[0] > rle_threshold).astype(np.uint8))

    # Build the result
    return {
        "rle": rle,
        "mask": mask,
    }


def image_file_generator(images_dir, images_csv=None):
    EXT = "tiff"

    # Load from images_dir
    if images_csv is None:
        for image_file in Path(images_dir).rglob(f"*.{EXT}"):
            yield image_file, None

    # Load from dataframe
    else:
        df = pd.read_csv(images_csv)
        for row in df.itertuples():
            image_file = osp.join(images_dir, f"{row.id}.{EXT}")
            if not osp.exists(image_file):
                print(f"Image {image_file} doesn't exist, skipping")
                continue

            yield image_file, row.organ


def main(
    model_file,
    images_dir,
    image_meter_scale,
    network_scale,
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
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        device = 'cuda'
    else:
        device = 'cpu'
    assert isinstance(tta, bool)
    experiment_dir = Path(model_file).parent.parent.parent
    if config_file is None:
        config_file = osp.join(experiment_dir, "src", "configs", "u.yaml")
    if output_dir is None:
        output_dir = osp.join(experiment_dir, "predicts")
    os.makedirs(output_dir, exist_ok=True)

    # Create TiffReader initializer
    TiffReader = partial(
        BatchedTiffReader,
        pad_ratio=pad_ratio,
        batch_size=batch_size,
    )

    # Create inferer
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

    for image_file, organ in tqdm(gen):
        # image_meter_scale should be image specific
        scale = image_meter_scale / network_scale
        log("SCALE", scale)
        scaled_block_size = int(round(block_size / scale))

        with TiffReader(image_file, scaled_block_size) as image_reader:
            image_result = infer_image(image_reader, inferer, scale, organ=organ, rle_threshold=threshold)

        if output_csv is not None:
            result.append({
                "image_filename": osp.basename(image_file),
                "rle": image_result["rle"],
            })

        if output_dir is not None:
            mask_output_file = osp.join(output_dir, osp.basename(image_file))
            mask = image_result["mask"] * 255
            save_tiff(mask_output_file, mask)

    if output_csv is not None:
        result = pd.DataFrame(result)
        result.to_csv(output_csv, index=False)


def config_args():
    parser = ArgumentParser()

    # Required args
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--image_meter_scale", type=float, required=True)
    parser.add_argument("--network_scale", type=float, required=True)

    # Optional args
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--pad_ratio", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--tta_merge_mode", type=str, default="mean")
    parser.add_argument("--images_csv", type=str, default=None)

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**config_args())
