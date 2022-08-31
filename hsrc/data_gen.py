import json
import os
from pathlib import Path

import cv2
import fire
import rasterio
import numpy as np
from tqdm.auto import tqdm

from sampler import GdalSampler, GridSampler
from rle2tiff import polys2mask, save_mask, create_masks
from split_gen import do_split


import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)



def cut(*args, mode='inst', **kwargs):
    if mode == 'inst':
        cutter(*args, **kwargs, sampler=GdalSampler)
    elif mode == 'grid':
        cutter(*args, **kwargs, sampler=GridSampler)


def cutter(img_fns, ann_fns, masks_fns, dst_path, cropsize, scale, sampler, mask_fill_percent=0, total=1e5):
    wh = (cropsize * scale, cropsize * scale)

    for i_fn, m_fn, a_fn in tqdm(zip(img_fns, masks_fns, ann_fns)):
        s = sampler(i_fn, m_fn, img_polygons_path=a_fn, img_wh=wh, shuffle=True)
        # s = sampler.GridSampler(i_fn, m_fn, wh)
        print('Cutting ', i_fn, len(s))

        base_name = i_fn.with_suffix('').name
        img_dir = dst_path / 'images'
        os.makedirs(str(img_dir), exist_ok=True)

        mask_dir = dst_path / 'masks'
        os.makedirs(str(mask_dir), exist_ok=True)

        for idx, (i, m) in enumerate(s):

            orig_name = (str(idx).zfill(6) + '.png')

            img_name = img_dir / (base_name + '_' + orig_name)
            mask_name = mask_dir / (base_name + '_' + orig_name)

            i = i.transpose(1, 2, 0)
            m = m.transpose(1, 2, 0)

            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            #m = m.repeat(3,-1).astype(np.uint8)
            # m = 255 * m.repeat(3, -1).astype(np.uint8)  # as our masks are one bit : 0-1
            if m.max() > 0:
                m = m / m.max() # 0-1
            if mask_fill_percent > 0 and m.mean() < mask_fill_percent:
                continue

            m = (m.squeeze()).astype(np.uint8)


            # i = cv2.resize(i, (wh[0] // SCALE, wh[1] // SCALE), interpolation=cv2.INTER_AREA)
            # m = cv2.resize(m, (wh[0] // SCALE, wh[1] // SCALE), interpolation=cv2.INTER_AREA)
            i = cv2.resize(i, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(img_name), i)
            cv2.imwrite(str(mask_name), m)

            if idx > total: break


# def cut_grid(imgs_path, masks_path, dst_path):
#     filt = partial(utils.filter_ban_str_in_name, bans=['-', '_ell'])
#     img_fns = sorted(utils.get_filenames(imgs_path, '*.tiff', filt))
#     masks_fns = sorted(utils.get_filenames(masks_path, '*.tiff', filt))

#     wh = (_base_wh * SCALE, _base_wh * SCALE)

#     print(img_fns, masks_fns)

#     for i_fn, m_fn, in tqdm(zip(img_fns, masks_fns)):
#         s = sampler.GridSampler(i_fn, m_fn, wh)

#         img_dir = dst_path / 'imgs' / i_fn.with_suffix('').name
#         os.makedirs(str(img_dir), exist_ok=True)

#         mask_dir = dst_path / 'masks' / i_fn.with_suffix('').name
#         os.makedirs(str(mask_dir), exist_ok=True)

#         for idx, (i, m) in enumerate(s):
#             if (i.mean() < 10 or (i.mean() > 205 and i.std() < 20) or
#                 ((i.mean(0) < 10).sum() > 0.1 * wh[0] * wh[1])) and (
#                     np.random.random() > .05):
#                 continue
#             orig_name = (str(idx).zfill(6) + '.png')
#             print(i.shape, i.mean(), i.std(),
#                   (i.mean(0) < 10).sum() / (wh[0] * wh[1]), orig_name)

#             img_name = img_dir / orig_name
#             mask_name = mask_dir / orig_name

#             i = i.transpose(1, 2, 0)
#             i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

#             m = m.transpose(1, 2, 0)
#             # m = 255 * m.repeat(3, -1).astype(np.uint8)
#             m = (255 * m.squeeze()).astype(np.uint8)

#             i = cv2.resize(i, (wh[0] // SCALE, wh[1] // SCALE), interpolation=cv2.INTER_AREA)
#             m = cv2.resize(m, (wh[0] // SCALE, wh[1] // SCALE), interpolation=cv2.INTER_NEAREST)

#             cv2.imwrite(str(img_name), i)
#             cv2.imwrite(str(mask_name), m)

#             if DEBUG and idx > TOTAL: break


def start(src, dst, src_scale, cropsize, total=1e6, recursive=False, ann_source='rle'):
    # HUBMAP_KIDNEY_SCALE = .55 #um; .5 or .65 to be accurate
    # HUBMAP_COLON_SCALE = ? # TODO i have no idea
    # GTEX_SCALE = .49
    MODE = 'grid'
    mask_fill_percent = 0.01
    ext = '*.tiff'

    HPA_SCALE = .4
    _base_scale = (3 * 1000 / 1024)
    scale = _base_scale * HPA_SCALE / src_scale

    src = Path(src)
    dst = Path(dst) / 'preprocessed'
    dst.mkdir(exist_ok=True, parents=True)

    _imgs = list(src.rglob(ext) if recursive else src.glob(ext))
    anns = []
    imgs = []
    for img in _imgs:
        ann_files = img.parent.glob(f"{img.stem}*.json")
        ann_files = sorted(list(ann_files))[::-1]
        if len(ann_files) == 0:
            continue
        ann = ann_files[0]
        imgs.append(img)
        anns.append(ann)

    print('Total images: ', len(imgs))

    masks_path = dst / 'bigmasks'  # will be created
    # if not masks_path.exists():
    if True:
        masks_path.mkdir(exist_ok=True)
        if ann_source == 'json':
            for img, ann in zip(imgs, anns):
                with open(str(ann), 'r') as f:
                    data = json.load(f)
                polys = []
                for rec in data:
                    poly = rec['geometry']['coordinates'][0]
                    polys.append(poly)
                h, w = rasterio.open(img).shape
                mask = polys2mask(polys, h, w)
                save_mask(mask, masks_path / img.name)

        elif ann_source == 'rle':
            create_masks(str(src.parent), str(masks_path))
    else:
        print('\n\nMASKS ALREADY CREATED? SKIPING BIG TIFF MASK CREATION')


    name = f'{scale:.3f}_{cropsize}'
    cut_path = dst / 'CUTS' / name
    masks = [masks_path / i.name for i in imgs]
    #if not cut_path.exists():
    cut(imgs, anns, masks, cut_path, cropsize, scale, total=total, mode=MODE, mask_fill_percent=mask_fill_percent)

    # do_split(glomi_path, dst / f'SPLITS/{name}', mode=MODE)

    # if not grid_path.exists(): cut_grid(imgs_path, masks_path, grid_path)
    # do_split(grid_path, dst / 'SPLITS/grid_split' )


if __name__ == '__main__':
    fire.Fire(start)
