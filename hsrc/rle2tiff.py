import argparse
from pathlib import Path

from rasterio.features import rasterize
from shapely.geometry import Polygon
from tqdm import tqdm
import pandas as pd
import numpy as np
import rasterio

import utils


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def convex2mask(poly, mask_shape):
    polygon = Polygon(poly)
    mask = rasterize([polygon], out_shape=mask_shape)
    return mask


def rasterpoly(poly):
    poly = np.array(poly)
    x,y = poly[:,0].min(), poly[:,1].min()
    w,h = poly[:,0].max() - x, poly[:,1].max() - y
    y,x,h,w = [int(i) for i in [y,x,h,w]]

    poly[:,0]-=x
    poly[:,1]-=y

    polygon = Polygon(poly)
    mask = rasterize([polygon], out_shape=(h,w))
    # print(mask.shape, mask.max())
    return y,x,h,w,mask


def raster_polys(polys, h, w):
    mask = np.zeros((h, w), dtype=np.float16)
    for poly in polys:
        y,x,h,w, poly_mask = rasterpoly(poly)
        mask[y:y+h, x:x+w] += poly_mask
    mask = np.clip(mask, 0, 1).astype(np.uint8)
    return mask


def polys2mask(polys, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        poly_mask = convex2mask(poly, (h, w))
        mask = np.maximum(mask, poly_mask)
    return mask


def rle2mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape)


def read_train_csv(path):
    return pd.read_csv(str(path))


def get_rle_by_name(name, csv_path):
    data = read_train_csv(csv_path)
    idx = int(data[data['id'] == name].index.values)
    rle = data.iloc[idx, 1]
    return rle


def save_mask(mask, path):
    h, w = mask.shape if len(mask.shape) == 2 else mask.shape[1:]
    assert mask.dtype == np.uint8
    dst = rasterio.open(path,
                        'w',
                        driver='GTiff',
                        height=h,
                        width=w,
                        count=1,
                        nbits=1,
                        dtype=np.uint8)
    dst.write(mask, 1)  # 1 band
    dst.close()
    del dst


def create_masks(root, dst_path):
    root = Path(root)
    dst_path = Path(dst_path)

    assert (root / 'train.csv').exists()

    tiff_files = list((root / 'train').glob('*.tiff'))
    for tiff_file in tqdm(tiff_files):
        _, (W, H), _ = utils.get_basics_rasterio(tiff_file)
        rle = get_rle_by_name(
            tiff_file.with_suffix('').name, root / 'train.csv')
        mask = rle2mask(rle, (H, W)).T
        print(tiff_file, W, H, mask.shape, mask.dtype, mask.max())
        save_mask(mask, dst_path / tiff_file.name)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--src", default='input/hm/', type=str)
#     parser.add_argument("--dst", default='input/bigmasks', type=str)
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     start(args.src, args.dst)
