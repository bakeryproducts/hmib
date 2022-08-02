import json
import random
import numpy as np
from pathlib import Path
from functools import partial
from dataclasses import dataclass, replace

import rasterio
import pandas as pd
from shapely.geometry import Polygon
from rasterio.features import rasterize

from functools import lru_cache

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


ORGANS = {k:i for i,k in enumerate(['prostate', 'spleen', 'lung', 'largeintestine', 'kidney'])}


def read_ann(p):
    with open(str(p), 'rb') as f:
        data = json.load(f)
    return data


def convert_ann(data):
    uni = None
    for d in data:
        p = Polygon(d)
        p = p.buffer(0)
        if uni is None:
            uni = p
        else:
            uni = uni.union(p)
    return uni


class Mult:
    def __init__(self, ds, rate):
        self.ds = ds
        self.rate = rate
        self.total = len(self.ds)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            if idx[-1] > self.__len__(): raise StopIteration
            idx = [i % self.total for i in idx]
        else:
            if idx > self.__len__(): raise StopIteration
            idx = idx % self.total
        return self.ds[idx]

    def __len__(self): return self.rate * len(self.ds)


@dataclass
class Label:
    lid: int
    fname: str
    organ: int
    data_source: str
    w: int
    h: int
    rle: str


def create_label(row):
    lid = row['id']
    organ_name = row['organ']
    organ = ORGANS[organ_name]

    l = Label(lid=lid,
              fname=f'{lid}.tiff',
              organ=organ,
              data_source=row['data_source'],
              w=row['img_width'],
              h=row['img_height'],
              rle=row['rle'],
             )
    return l


def read_meta(df):
    labels = []
    for i, row in df.iterrows():
        label = create_label(row)
        labels.append(label)
    return labels


class TiffImage:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        fd = rasterio.open(self.file)
        return fd

    @lru_cache(None)
    def load(self):
        img_data = self().read()
        if len(img_data.shape) < 3:
            img_data = np.expand_dims(img_data, 0)
        img_data = img_data.transpose(1,2,0) # H,W,C
        return img_data

    def randcrop(self, cropsize):
        fd = rasterio.open(self.file)
        H,W = fd.shape
        h,w = cropsize
        x,y = random.randint(0,W-w), random.randint(0,H-h)
        return self.crop(y,x,h,w), (y,x,h,w)

    def crop(self, y,x,h,w):
        fd = rasterio.open(self.file)
        window=rasterio.windows.Window(x, y, w, h)
        return fd.read((1,2,3), window=window)

    @property
    def shape(self):
        fd = rasterio.open(self.file)
        return fd.shape


class CropDataset:
    def __init__(self, img_ds, ann_ds, cropsize):
        self.ids = img_ds
        self.ads = ann_ds
        self.cropsize = cropsize

    def __getitem__(self, idx):
        i = self.ids[idx]
        a = self.ads[idx]

        H,W = i.shape
        # assert shapes
        h,w = self.cropsize
        x,y = random.randint(0,W-w), random.randint(0,H-h)

        ic = i.crop(y,x,h,w)
        ac = a.crop(y,x,h,w)
        return ic, ac

    def __len__(self): return len(self.ds)


class JsonAnnotations:
    def __init__(self, ann_root):
        self.ann_root = Path(ann_root)

    def __call__(self, label):
        data = read_ann((self.ann_root / label.fname).with_suffix('.json'))
        poly = convert_ann(data)
        mask = rasterize([poly], out_shape=(label.h, label.w))
        mask = mask * 255
        mask = np.expand_dims(mask, -1)
        return mask


class TiffImages:
    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, label):
        path = self.root / label.fname
        i = TiffImage(path)
        i = i.load()
        # print(i.shape, i.max(), i.dtype)
        return i


class MaskImages:
    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, label):
        path = self.root / label.fname
        i = TiffImage(path)
        i = i.load() * 255
        return i


class BlankImages:
    def __call__(self, label):
        return 0


class Names:
    def __init__(self, root, suffix=None):
        self.root = Path(root)
        self.suffix = suffix

    def __call__(self, label):
        path = self.root / label.fname
        if self.suffix is not None:
            path = path.with_suffix(self.suffix)
        return str(path)


class DataPair:
    def __init__(self, imgs, anns):
        self.anns = anns
        self.imgs = imgs

    def __call__(self, label):
        img_data = self.imgs(label)
        ann_data = self.anns(label)
        return dict(fname=label.fname, x=img_data, y=ann_data, cls=label.organ)


class DfDataset:
    def __init__(self, data, base_df, ind_df=None, index_paths=None):
        if index_paths:
            ind_df = load_index_df(index_paths)
        inddf = ind_df.reset_index(drop=True)

        df = base_df.iloc[inddf[0].values]
        self.labels = read_meta(df)
        self.data = data

    def __getitem__(self, idx):
        label = self.labels[idx]
        return self.data(label)

    def __len__(self): return len(self.labels)


class ExtDfDataset:
    def __init__(self, data, base_df, ind_df=None, index_paths=None):
        if index_paths:
            ind_df = load_index_df(index_paths)
        inddf = ind_df.reset_index(drop=True)

        df = base_df.iloc[inddf[0].values]
        self.labels = read_meta(df)
        ll = []
        for l in self.labels:
            for i in range(9):
                lc = replace(l) # copy dataclass
                name, ext = lc.fname.split('.')
                lc.fname = f'{name}_{i}.{ext}'
                ll.append(lc)
        self.labels = ll
        self.data = data

    def __getitem__(self, idx):
        label = self.labels[idx]
        return self.data(label)

    def __len__(self): return len(self.labels)


def load_index_df(index_paths):
    dfs = []
    for p in index_paths:
        df = pd.read_csv(str(p), header=None)
        dfs.append(df)

    if len(dfs) > 1: df = pd.concat(dfs)
    else: df = dfs[0]
    return df


class MainDataset:
    def __init__(self,
                 cfg,
                 root,
                 ann_path,
                 base_path,
                 index_paths,
                 train,
                 ImgLoader,
                 AnnLoader,
                 **kwargs):
        assert all([p.exists() for p in index_paths]), index_paths

        imgs = ImgLoader(root)
        anns = AnnLoader(ann_path)
        data = DataPair(imgs, anns)

        base_df = pd.read_csv(base_path)
        ind_df = load_index_df(index_paths)

        rate = kwargs.pop('rate')
        # ds = DfDataset(data=data, base_df=base_df, ind_df=ind_df, **kwargs)
        ds = ExtDfDataset(data=data, base_df=base_df, ind_df=ind_df, **kwargs)
        self.ds = Mult(ds, rate)

    def __len__(self): return len(self.ds)
    def __getitem__(self, *args, **kwargs): return self.ds.__getitem__(*args, **kwargs)



def item_reader(item):
    i, a = item
    return i, a


class MaskGenerator:
    def __init__(self,
                 input_size,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=.3):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        self.init_params()

    def init_params(self):
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))


    def __call__(self, progress=None):
        if progress is not None:
            self.mask_ratio = min(progress, .7)
            self.init_params()

        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        # mask[5:7,5:7] = 1
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask
