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

from mp import parallel_read
NUM_PROCESSES = 8

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def read_tiff(name):
    #return rio.open(str(name)).read()[0]
    return parallel_read(name, NUM_PROCESSES)[0]


def mask_to_poly(mask, tolerance, minarea, min_sieve_pix=25):
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
            if shp.area > minarea:
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

    #postfix = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
    postfix = 'AUTO'
    fn = dst / (name + "|" + postfix + '.json')
    print(f'Saving annotations at {fn}')

    with open(str(fn), 'w') as f:
        json.dump(fixed_polys, f, indent=4)




def process_mask(name, dst, tolerance, threshold, minarea):
    mask = read_tiff(name)
    # print(mask.shape, mask.max())
    mask = mask > threshold
    mask = mask.astype(np.uint8)
    if mask.max() <= 1: mask = mask * 255
    polys = mask_to_poly(mask, tolerance, minarea)
    create_ann(name.stem, dst, polys)


def start(name, dst=None, mode='S', ext='tiff', tolerance=3, minarea=1, threshold=127):
    name = Path(name)
    if dst is not None: dst = Path(dst)
    if mode == 'S':
        print('\t Single file mode')
        if dst is None: dst = name.parent.parent / 'polys'
        dst.mkdir(exist_ok=True)
        process_mask(name, dst, tolerance, threshold, minarea)

    elif mode == 'F':
        print('\t Folder mode')
        if dst is None: dst = name.parent / 'polys'
        dst.mkdir(exist_ok=True)
        masks = list(name.rglob(f'*.{ext}'))
        print('\t Masks: ', len(masks))

        for fn in tqdm(masks):
            print(f'\t\t Doing {fn}')
            d = dst / fn.parent.name
            d.mkdir(exist_ok=True)
            process_mask(fn, d, tolerance, threshold, minarea)


if __name__ == '__main__':
    fire.Fire(start)
