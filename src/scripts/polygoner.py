#!/usr/bin/python3

import json
import datetime
from pathlib import Path

import fire
import numpy as np
import rasterio as rio
from tqdm import tqdm
from rasterio import features
from shapely import geometry


def mask_to_poly(mask, tolerance, min_sieve_pix=25):
    mask = features.sieve(mask, min_sieve_pix)
    pp = []
    for r, v in features.shapes(mask, mask > 0):
        cds = r['coordinates']
        #print(v, len(cds))
        if len(cds) > 1:
            #raise Exception
            cds = cds[0:1]
        if v > 0:
            poly = np.array(cds)[0]
            shp = geometry.Polygon(poly)
            poly_s = shp.simplify(tolerance=tolerance)
            if shp.area > 1:
                poly = np.array(poly_s.exterior.xy).T
                pp.append(poly)
    return pp


def create_ann(name, dst, polys):
    fixed_polys = []
    for i, poly in enumerate(polys):
        if isinstance(poly, np.ndarray):
            poly = poly.tolist()
        poly = dict(geometry=dict(coordinates=[poly]))
        fixed_polys.append(poly)

    timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
    fn = dst / (name + "|" + timestamp + '.json')
    print(f'Saving annotations at {fn}')

    with open(str(fn), 'w') as f:
        json.dump(fixed_polys, f, indent=4)


def read_tiff(name):
    return rio.open(str(name)).read()[0]


def process_mask(name, dst, tolerance, threshold):
    mask = read_tiff(name)
    mask = mask > threshold
    mask = mask.astype(np.uint8)
    if mask.max() <= 1: mask = mask * 255
    polys = mask_to_poly(mask, tolerance)
    create_ann(name.stem, dst, polys)


def start(name, dst=None, mode='S', ext='tiff', tolerance=3, threshold=127):
    name = Path(name)
    if dst is not None: dst = Path(dst)
    if mode == 'S':
        print('\t Single file mode')
        if dst is None: dst = name.parent.parent / 'polys'
        dst.mkdir(exist_ok=True)
        process_mask(name, dst, tolerance, threshold)

    elif mode == 'F':
        print('\t Folder mode')
        if dst is None: dst = name.parent / 'polys'
        dst.mkdir(exist_ok=True)
        masks = list(name.rglob(f'*.{ext}'))
        print('\t Masks: ', len(masks))

        for fn in tqdm(masks):
            print(f'\t\t Doing {fn}')
            process_mask(fn, dst, tolerance, threshold)


if __name__ == '__main__':
    fire.Fire(start)
