import json
import os
from pathlib import Path

import cv2
import fire
import rasterio
import numpy as np
from tqdm.auto import tqdm

from sampler import GdalSampler, GridSampler
from rle2tiff import save_mask, create_masks, raster_polys
#from split_gen import do_split


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

            i = cv2.resize(i, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(str(img_name), i)
            cv2.imwrite(str(mask_name), m)

            if idx > total: break



def start(src, dst, src_scale, dst_scale, cropsize, total=1e6, mode='grid', recursive=False, ext='tiff', HACKHPA=False):
    #dst_scale = .4 * 3
    # mode == [inst, grid]

    MODE = mode
    mask_fill_percent = 0.01
    ext = f'*.{ext}'

    scale = dst_scale / src_scale

    src = Path(src)
    dst = Path(dst)
    dst.mkdir(exist_ok=True, parents=True)

    _imgs = list(src.rglob(ext) if recursive else src.glob(ext))
    anns = []
    imgs = []
    for img in _imgs:
        ann_files = img.parent.glob(f"{img.stem}|MANUAL.json")
        # ann_files = img.parent.glob(f"{img.stem}*.json")
        ann_files = sorted(list(ann_files))[::-1]
        if len(ann_files) == 0:
            continue
        ann = ann_files[0]
        imgs.append(img)
        anns.append(ann)

    print('Total images: ', len(imgs))

    ############################# CREATING MASKS FROM ANNOTATIONS
    ann_source = 'json'
    masks_path = dst / 'bigmasks'  # will be created

    if not masks_path.exists():
    # if True:
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
                print(f'CONVERTING POLYGONS {img}, {len(polys)}')
                mask = raster_polys(polys, h, w)
                save_mask(mask, masks_path / img.name)

        elif ann_source == 'rle':
            create_masks(str(src.parent), str(masks_path))
    else:
        print('\n\nMASKS ALREADY CREATED? SKIPING BIG TIFF MASK CREATION')
    ############################# CREATING MASKS FROM ANNOTATIONS

    if HACKHPA:
        #import shutil
        name = f'{scale:.3f}'
        cut_path = dst / name

        img_dir = cut_path / 'images'
        os.makedirs(str(img_dir), exist_ok=True)
        mask_dir = cut_path / 'masks'
        os.makedirs(str(mask_dir), exist_ok=True)
        for img_name in imgs:
            #print(fm, mask_dir)
            #shutil.move(str(fm), str(mask_dir))
            # we need to get pngs(
            new_name = img_name.stem + '.png'

            img = cv2.imread(str(img_name), cv2.IMREAD_UNCHANGED)
            fix_img_name = img_dir / new_name
            cv2.imwrite(str(fix_img_name), img)

            mask_name = masks_path / img_name.name
            mask = cv2.imread(str(mask_name), cv2.IMREAD_UNCHANGED)
            mask = mask / 255
            #print(mask_name, mask.max())
            fix_mask_name = mask_dir / new_name
            cv2.imwrite(str(fix_mask_name), mask)
            #shutil.copy(str(i), str(img_dir))
        return



    ################# CUTTING TO PIECES
    name = f'{scale:.3f}_{cropsize}'
    cut_path = dst / name
    masks = [masks_path / i.name for i in imgs]
    #if not cut_path.exists():
    cut(imgs, anns, masks, cut_path, cropsize, scale, total=total, mode=MODE, mask_fill_percent=mask_fill_percent)


if __name__ == '__main__':
    fire.Fire(start)
