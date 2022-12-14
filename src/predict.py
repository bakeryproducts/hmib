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
import mp
from tiff import save_tiff
from mask_utils import rle_encode
from mp import parallel_block_read
from block_utils import generate_block_coords, crop, paste_crop, pad_block, batcher

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def select_organ_from_predict(yb, organ, organs):
    # yb BCHW
    if organ and organ != 'none':
        yb = yb[:, organs[organ]].unsqueeze(1)
    else:
        yb, _ = torch.max(yb, dim=1, keepdim=True)
    return yb


def extend_input_organ(xb, organ, organs):
    b,_,h,w = xb.shape
    cls = torch.tensor(organs[organ]).repeat(b)

    cls_layer = torch.ones(b,1,h,w).float()
    cls_layer = cls_layer * cls.view(-1, 1,1,1)
    cls_layer = (cls_layer.to(xb) + 1) / 5
    xb = torch.hstack([xb, cls_layer])
    return xb


def log(*m):
    # TODO normal logging?
    pass
    # print(m)


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


def infer_whole(name, inferer, scale, pad_size, preprocess, postprocess, **kwargs):
    interpolation_mode='bilinear'

    img = rio.open(name).read()

    c,h,w = img.shape

    output_h = int(np.ceil(h * scale / 32)) * 32
    output_w = int(np.ceil(w * scale / 32)) * 32

    img = torch.from_numpy(img).unsqueeze(0).float()
    img = torch.nn.functional.interpolate(img, (output_h, output_w), mode=interpolation_mode)

    img = inferer.preprocess(img, preprocess=preprocess)
    pred = inferer(img)  # bchw, logit

    mask = postprocess(pred)
    mask = torch.nn.functional.interpolate(mask, (h, w), mode=interpolation_mode)
    mask.sigmoid_()
    mask = mask.cpu()
    mask = mask[0]
    return mask



def infer_whole_with_blocks(name, inferer, scale, pad_size, block_size, preprocess, postprocess, use_mp=False, **kwargs):
    interpolation_mode='bilinear'
    if use_mp: img = mp.parallel_read(name, 8)
    else: img = rio.open(name).read()

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
        batch_blocks = inferer.preprocess(part.unsqueeze(0), preprocess=preprocess)
        batch_masks = inferer(batch_blocks)  # bchw, logit
        paste_crop(pred, batch_masks[0], cd, pad_size)

    pred = pred.unsqueeze(0)
    # mask = select_organ_from_predict(pred, organ, ORGANS)
    mask = postprocess(pred)
    mask = torch.nn.functional.interpolate(mask, (h, w), mode=interpolation_mode)
    mask.sigmoid_()
    mask = mask.cpu()
    mask = mask[0]
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
        device = 'cuda'
    else:
        device = 'cpu'

    if tta != 'none':
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
    tta='none',
    device=None,
    tta_merge_mode="mean",
    images_csv=None,
    ext='tiff',
    scale_block=True,
    use_mp=False,
):
    model_file = Path(model_file)
    experiment_dir = model_file.parent.parent.parent
    result_dir = experiment_dir / f'{tta}_predicts'
    if output_dir is not None: result_dir = result_dir / output_dir
    if result_dir.exists():
        print(f'{result_dir} exists! quitting !')
        return
    result_dir.mkdir(exist_ok=True, parents=True)

    config_file = experiment_dir / 'src/configs/u.yaml'
    # config_file = experiment_dir / 'src/configs/lung.yaml'
    inferer = init_infer(model_file, config_file, device, tta, tta_merge_mode)

    if 'ORGANS' not in inferer.cfg.DATA:
        organs = ['prostate', 'spleen', 'lung', 'largeintestine', 'kidney']
        ORGANS = {k:i for i,k in enumerate(organs)}
        REV_ORGANS = {v:k for k,v in ORGANS.items()}
        inferer.cfg.DATA.ORGANS = ORGANS

    if organ == 'colon': # synonim
        organ = 'largeintestine'

    result = []
    images_dir = Path(images_dir)
    gen = image_file_generator(images_dir, images_csv, ext)


    for image_file, _ in tqdm(gen):
        print('\n \t', image_file)
        scale = image_meter_scale / network_scale

        block_size = int(round(base_block_size / scale)) if scale_block else base_block_size
        pad_size = int(block_size * pad) if pad <= 1.0 else pad

        img_size = rio.open(image_file).shape # well, small price for func reader
        dst = result_dir / image_file.parent.stem
        dst.mkdir(exist_ok=True)
        log("SCALE", scale)

        if mode == "whole_image":
            mask = infer_whole(image_file,
                               inferer,
                               scale,
                               pad_size,
                               preprocess=partial(extend_input_organ, organ=organ, organs=inferer.cfg.DATA.ORGANS),
                               postprocess=partial(select_organ_from_predict, organ=organ, organs=inferer.cfg.DATA.ORGANS),
                               )
        elif mode == "whole_blocks":
            mask = infer_whole_with_blocks(
                image_file,
                inferer,
                scale,
                pad_size,
                block_size=block_size,
                preprocess=partial(extend_input_organ, organ=organ, organs=inferer.cfg.DATA.ORGANS),
                postprocess=partial(select_organ_from_predict, organ=organ, organs=inferer.cfg.DATA.ORGANS),
                use_mp=use_mp,
            )
        else:
            # mode not in modes
            raise ValueError


        if output_csv:
            result.append({
                "image_filename": image_file.name,
                "rle": rle_encode((mask[0] > threshold).astype(np.uint8))
            })

        if result_dir:
            save_tiff(dst / image_file.name, mask * 255)

    if output_csv:
        result = pd.DataFrame(result)
        result.to_csv(output_csv, index=False)


if __name__ == "__main__":
    fire.Fire(main)
