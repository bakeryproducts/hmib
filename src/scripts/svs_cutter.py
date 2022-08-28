#!/usr/bin/python3
import json

import fire
import slideio
import rasterio
import numpy as np
from tqdm import tqdm
from pathlib import Path



def read_svs(p, x,y,h,w):
    slide = slideio.open_slide(str(p), 'SVS')
    scene = slide.get_scene(0)
    image = scene.read_block((x,y,h,w))
    return image


def save_tiff(dst_tiff, image, h, w):
    profile = rasterio.profiles.default_gtiff_profile
    profile.update({
        'height': h,
        'width': w,
        'count': 3,
    })
    with rasterio.open(dst_tiff, 'w', **profile) as f:
        f.write(image.transpose(2,0,1))


def do_cut(src_ann):
    # be sure, that annotation is body type, ~ 1- 10 huge polys
    assert 'BODY' in src_ann
    # src_ann = Path('../input/extra/gtex/images/colon/GTEX-1K2DA-2026|BODY.json')
    src_ann = Path(src_ann)
    assert src_ann.exists(), src_ann
    name = src_ann.stem.split('|')[0]
    src = src_ann.parent / (name + '.svs')
    assert src.exists(), src
    dst = src_ann.parent.parent.parent / 'pieces' / src_ann.parent.name / name
    dst.mkdir(exist_ok=True, parents=True)

    with open(str(src_ann), 'r') as f:
        data = json.load(f)
    polys = [p['geometry']['coordinates'][0] for p in data]

    for i, p in tqdm(enumerate(polys)):
        p = np.array(p)
        x, y, x2, y2 = p[:,0].min(), p[:,1].min(), p[:,0].max(), p[:,1].max()
        w,h = x2 - x, y2 - y
        x,y,w,h = [int(j) for j in [x,y,w,h]]
        image = read_svs(src,x,y,w,h)

        dst_tiff = dst / (f'{i}_{x}_{y}_{w}_{h}.tiff')
        print(f'Saving {dst_tiff}')
        save_tiff(dst_tiff, image, h, w)


if __name__ == '__main__':
    fire.Fire(do_cut)
