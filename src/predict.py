import os
import os.path as osp
from functools import partial
from pathlib import Path

#import albumentations.augmentations.geometric.functional as AGF
import fire
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from block_utils import paste_crop
from infer import Inferer
from mask_utils import rle_encode
from tiff import BatchedTiffReader, save_tiff


def select_organ_from_predict(yb, organ=None):
    # yb BCHW
    # copy ORGANS from data, better not to source data.py
    ORGANS = {k:i for i,k in enumerate(['prostate', 'spleen', 'lung', 'largeintestine', 'kidney'])}
    if organ is not None: yb = yb[:, ORGANS[organ]][:, None]
    else: yb, _ = torch.max(yb, dim=1, keepdim=True)
    return yb


def log(*m):
    pass
    #print(m)


def infer_image(image_reader, inferer, rle=True, organ=None):
    H, W = image_reader.shape

    img_scale = image_reader.image_meter_scale # .4
    network_scale = .4 * 3# * 1 / (1024/3000)
    s = img_scale / network_scale
    pad_size = image_reader.scaled_pad_size
    # Infer batch by batch
    mask = np.zeros((1, H, W), dtype=float)#.astype(bool if rle else np.float32)
    if len(image_reader) > 20: image_reader = tqdm(image_reader)
    for batch_blocks, batch_coords in image_reader:
        # BCHW
        # Infer batch
        log('LOAD', batch_blocks.shape, batch_coords[0])
        batch_blocks = torch.from_numpy(batch_blocks)
        batch_blocks = torch.nn.functional.interpolate(batch_blocks, scale_factor=(s, s))
        log('INFER', batch_blocks.shape, batch_blocks.max())
        batch_masks = inferer(batch_blocks) # bchw, logit
        log('PREDICT', batch_masks.shape, batch_masks.max())
        batch_masks = select_organ_from_predict(batch_masks, organ)
        batch_masks = torch.nn.functional.interpolate(batch_masks, scale_factor=(1/s, 1/s))
        batch_masks.sigmoid_()
        log('ORGAN', batch_masks.shape,)
        #batch_masks = batch_masks.permute(0,2,3,1) # bhwc
        batch_masks = batch_masks.cpu()
        log('FUNAL', batch_masks.shape)


        # Copy block mask to the original
        for block_mask, block_cd in zip(batch_masks, batch_coords):
            # no way
            #block_mask = AGF.scale(block_mask, image_reader.inv_network_scale)
            # block_mask = block_mask.transpose((2, 0, 1))
            log(block_mask.shape, block_mask.max(), block_cd)
            paste_crop(mask, block_mask, block_cd, pad_size)

    log('MASK', mask.shape, mask.mean(), mask.max())

    # Build the result
    return {
        "rle": '',#rle_encode(mask),
        "mask": mask#.astype(np.uint8).transpose((1, 2, 0)),
    }


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
    model_file,
    images_dir,
    image_meter_scale,
    output_dir=None,
    output_csv=None,
    config_file=None,
    block_size=512,
    network_scale=1024/3000,
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
        block_size=block_size,
        image_meter_scale=image_meter_scale,
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
    for image_file, image_id, organ in tqdm(image_file_generator(images_dir, images_csv)):
        with TiffReader(image_file) as image_reader:
            image_result = infer_image(image_reader, inferer, rle=True, organ=organ)

        if output_csv is not None:
            result.append({
                "id": image_id,
                "rle": image_result["rle"],
            })

        if output_dir is not None:
            mask_output_file = osp.join(output_dir, osp.basename(image_file))
            mask = image_result["mask"] * 255
            save_tiff(mask_output_file, mask)

    if output_csv is not None:
        result = pd.DataFrame(result)
        result.to_csv(output_csv, index=False)


if __name__ == "__main__":
    fire.Fire(main)
