from pathlib import Path
from functools import partial

import fire
import torch
import ttach
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio

import infer
from data import ORGANS
from tiff import save_tiff
from mask_utils import rle_encode
from mp import parallel_block_read
from block_utils import generate_block_coords, crop, paste_crop, pad_block, batcher

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def select_organ_from_predict(yb, organ):
    # yb BCHW
    # copy ORGANS from data, better not to source data.py
    if organ and organ != 'none':
        yb = yb[:, ORGANS[organ]].unsqueeze(1)
    else:
        yb, _ = torch.max(yb, dim=1, keepdim=True)
    return yb


def extend_input_organ(xb, organ):
    b,_,h,w = xb.shape
    cls = torch.tensor(ORGANS[organ]).repeat(b)

    cls_layer = torch.ones(b,1,h,w).float()
    cls_layer = cls_layer * cls.view(-1, 1,1,1)
    cls_layer = (cls_layer.to(xb) + 1) / 5
    xb = torch.hstack([xb, cls_layer])
    return xb


def log(*m):
    # TODO normal logging?
    pass
    # print(m)


def infer_whole(name, inferer, scale, pad_size, image_size, organ=None, interpolation_mode='bilinear', extra_postprocess=lambda x:x):
    img = rio.open(name).read()
    c,h,w = img.shape

    output_h = int(np.ceil(h * scale / 32)) * 32
    output_w = int(np.ceil(w * scale / 32)) * 32

    img = torch.from_numpy(img).unsqueeze(0).float()
    img = torch.nn.functional.interpolate(img, (output_h, output_w), mode=interpolation_mode)

    img = inferer.preprocess(img, postp=extra_postprocess)
    pred = inferer(img)  # bchw, logit

    mask = select_organ_from_predict(pred, organ)
    mask = torch.nn.functional.interpolate(mask, (h, w), mode=interpolation_mode)
    mask.sigmoid_()
    mask = mask.cpu()
    mask = mask[0]
    return mask


def pad_image_block_size(img, bs):
    _, H, W = img.shape
    px = W - bs * (W // bs)
    if px > 0: px = bs-px
    py = H - bs * (H // bs)
    if py > 0: py = bs-py
    # pimg = torch.nn.functional.pad(img, pad=(0,px,0,py), mode='reflect')
    pimg = torch.nn.functional.pad(img, pad=(0,px,0,py), mode='constant')
    return pimg


def cropper(image, bs, pad):
    C,H,W = image.shape
    padimg = torch.nn.functional.pad(image, pad=(pad,pad,pad,pad), mode='constant')
    cds = generate_block_coords(H,W,(bs,bs))
    for cd in cds:
        y,x,h,w = pad_block(*cd, pad)
        padcd = y+pad, x+pad, h, w
        c = crop(padimg, *padcd).clone()
        y, x, h, w = cd
        h, w = min(h, H - y), min(w, W - x)
        cd = y,x,h,w
        yield c, cd


def infer_whole_with_blocks(name, inferer, scale, pad_size, image_size, block_size, organ=None, interpolation_mode='bilinear', extra_postprocess=lambda x:x):
    img = rio.open(name).read()
    c,h,w = img.shape

    output_h = int(np.ceil(h * scale / 32)) * 32
    output_w = int(np.ceil(w * scale / 32)) * 32

    img = torch.from_numpy(img).unsqueeze(0).float()
    img = torch.nn.functional.interpolate(img, (output_h, output_w), mode=interpolation_mode)

    orig_image = img[0]
    _, orig_H, orig_W = orig_image.shape
    image = pad_image_block_size(orig_image, block_size)

    # pred = torch.zeros_like(orig_image)
    pred = torch.zeros((5, orig_H, orig_W), dtype=float)
    gen = cropper(image, block_size, pad_size)
    for part, cd in gen:
        batch_blocks = inferer.preprocess(part.unsqueeze(0), postp=extra_postprocess)
        batch_masks = inferer(batch_blocks)  # bchw, logit
        paste_crop(pred, batch_masks[0], cd, pad_size)

    pred = pred.unsqueeze(0)
    mask = select_organ_from_predict(pred, organ)
    mask = torch.nn.functional.interpolate(mask, (h, w), mode=interpolation_mode)
    mask.sigmoid_()
    mask = mask.cpu()
    mask = mask[0]
    return mask


def infer_image(dataloader, inferer, scale, pad_size, image_size, organ=None, interpolation_mode='bilinear', extra_postprocess=lambda x:x):
    H, W = image_size

    # Infer batch by batch
    mask = np.zeros((1, H, W), dtype=float)
    for batch_blocks, batch_coords in dataloader:
        batch_blocks = torch.from_numpy(np.stack(batch_blocks))
        batch_blocks = batch_blocks.cuda().float()

        # Fix scale to output_size be divisible by 32
        _, _, h, w = batch_blocks.shape
        output_h = int(np.ceil(h * scale / 32)) * 32
        output_w = int(np.ceil(w * scale / 32)) * 32

        # BCHW
        # Infer batch
        log('LOAD', batch_blocks.shape, batch_coords[0])
        batch_blocks = torch.nn.functional.interpolate(batch_blocks, (output_h, output_w), mode=interpolation_mode)
        #print(h, w, batch_blocks.shape)

        log('PREEXTEND', batch_blocks.shape)
        batch_blocks = inferer.preprocess(batch_blocks.float(), postp=extra_postprocess)
        log('INFER', batch_blocks.shape)
        batch_masks = inferer(batch_blocks)  # bchw, logit

        log('PREDICT', batch_masks.shape, batch_masks.max())
        batch_masks = select_organ_from_predict(batch_masks, organ)
        batch_masks = torch.nn.functional.interpolate(batch_masks, (h, w), mode=interpolation_mode)
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


def image_file_generator(images_dir, images_csv=None, ext='tiff'):
    # Load from images_dir
    if images_csv is None:
        for image_file in images_dir.rglob(f"*.{ext}"):
            yield image_file, None
    # Load from dataframe
    else:
        df = pd.read_csv(images_csv)
        for row in df.itertuples():
            image_file = images_dir / f"{row.id}.{ext}"
            if not image_file.exists():
                print(f"Image {image_file} doesn't exist, skipping")
                continue
            yield image_file, row.organ



def init_infer(model_file, config_file, device, tta, tta_merge_mode):
    if device is not None:
        # wont init all gups
        #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
        device = 'cuda'
    else:
        device = 'cpu'

    if tta is not None:
        transforms = []
        for t in tta.split('_'):
            if t == 'd4':
                transforms.extend([ttach.HorizontalFlip(), ttach.Rotate90(angles=[0, 90, 180, 270]),])
            elif t == 'scale':
                transforms.append(infer.ScaleStep([.8, 1, 1.2]))
            elif t == 'flip':
                transforms.extend([ttach.HorizontalFlip(), ttach.VerticalFlip(),])
            elif t == 'rotate':
                transforms.append(ttach.Rotate90(angles=[0, 90, 180, 270]),)
            elif t == 'stain':
                transforms.append(infer.StainTta(),)
            else:
                raise NotImplementedError
        tta_transforms = ttach.Compose(transforms)
    else:
        tta_transforms = None

    #Create inferer
    experiment_dir = model_file.parent.parent.parent
    inferer = infer.Inferer.create(
        model_file,
        config_file,
        experiment_dir,
        device=device,
        tta_transforms=tta_transforms,
        tta_merge_mode=tta_merge_mode,
    )
    return inferer


def main(
    model_file,
    images_dir,
    image_meter_scale,
    network_scale,
    organ,
    base_block_size,
    mode,
    output_dir=None,
    output_csv=None,
    config_file=None,
    pad=0.25,
    batch_size=4,
    threshold=0.5,
    tta=None,
    device=None,
    tta_merge_mode="mean",
    images_csv=None,
    ext='tiff',
    scale_block=True,
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
        pad: float, default 0.25
            if <1 Ratio of padding during the inference
            if >1 pad size
        batch_size: int, default 4
            Batch size
        tta: str
            modes : 'd4', 'd4_scale'
        device: int, default None
            Device for inference, should be 0, 1, ..  or None for cpu
        tta_merge_mode: str
            One of [mean, max]
        images_csv: str, default None
            Path to .csv file with images metadata
            If you pass this path only images from this df will be
            used for inference
    """
    model_file = Path(model_file)
    experiment_dir = model_file.parent.parent.parent
    if output_dir is None: output_dir = experiment_dir / "predicts"
    output_dir.mkdir(exist_ok=True)

    config_file = experiment_dir / 'src/configs/u.yaml'
    inferer = init_infer(model_file, config_file, device, tta, tta_merge_mode)

    result = []
    images_dir = Path(images_dir)
    gen = image_file_generator(images_dir, images_csv, ext)


    for image_file, _ in tqdm(gen):
        print('\n \t', image_file)
        scale = image_meter_scale / network_scale

        block_size = int(round(base_block_size / scale)) if scale_block else base_block_size
        pad_size = int(block_size * pad) if pad <= 1.0 else pad

        img_size = rio.open(image_file).shape # well, small price for func reader
        dst = output_dir / image_file.parent.stem
        dst.mkdir(exist_ok=True)
        log("SCALE", scale)

        if mode == "whole_image":
            mask = infer_whole(image_file,
                               inferer,
                               scale,
                               pad_size,
                               img_size,
                               organ=organ,
                               extra_postprocess=partial(extend_input_organ, organ=organ),
                               )
        elif mode == "whole_blocks":
            mask = infer_whole_with_blocks(
                image_file,
                inferer,
                scale,
                pad_size,
                img_size,
                block_size=block_size,
                organ=organ,
                extra_postprocess=partial(extend_input_organ, organ=organ),
            )
        elif mode == "part_blocks":
            _image_reader = parallel_block_read(image_file, block_size, pad_size, num_processes=8)
            image_reader = batcher(_image_reader, batch_size)
            mask = infer_image(image_reader,
                               inferer,
                               scale,
                               pad_size,
                               img_size,
                               organ=organ,
                               extra_postprocess=partial(extend_input_organ, organ=organ),
                               )
        else:
            # mode not in modes
            raise ValueError


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
