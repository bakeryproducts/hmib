import cv2
import json
import random
import numpy as np
from pathlib import Path
from functools import partial, lru_cache
from dataclasses import dataclass, replace

import rasterio
import pandas as pd
from rasterio.features import rasterize


import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


DOMAINS = {k:v for v, k in enumerate(['hpa', 'hubmap', 'gtex', 'undef'])}
ORGANS = {k:i for i,k in enumerate(['prostate', 'spleen', 'lung', 'largeintestine', 'kidney'])}
REV_ORGANS = {v:k for k,v in ORGANS.items()}


def read_ann(p):
    with open(str(p), 'rb') as f:
        data = json.load(f)
    return data


def convert_ann(data, fixer):
    uni = None
    for d in data:
        p = fixer(d)
        if p is None:
            continue
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
    def load(self, *args, **read_kwargs):
        img_data = self().read(**read_kwargs)
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


class PngImages:
    def __init__(self, root, suffix=None):
        self.root = Path(root)
        self.suffix = suffix

    def __call__(self, label):
        path = self.root / label.fname
        if self.suffix is not None:
            path = path.with_suffix(self.suffix)
        i = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        assert i is not None, path
        if len(i.shape) < 3:
            i = np.expand_dims(i, -1)
        else:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
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
        return dict(fname=label.fname, x=img_data, y=ann_data, cls=label.organ, dom=label.data_source)


class DfDataset:
    def __init__(self, data, base_df, ind_df=None, index_paths=None, overlap=0):
        if index_paths:
            ind_df = load_index_df(index_paths)
        inddf = ind_df.reset_index(drop=True)

        df = base_df.iloc[inddf[0].values]
        self.labels = read_meta(df)
        if overlap: self.labels = self.convert_to_overlap(self.labels, overlap)
        self.data = data

    def convert_to_overlap(self, labels, overlap):
        ll = []
        for l in labels:
            for i in range(overlap):
                lc = replace(l) # copy for dataclass
                name, ext = lc.fname.split('.')
                lc.fname = f'{name}_{i}.{ext}'
                ll.append(lc)
        return ll


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
        ds = DfDataset(data=data, base_df=base_df, ind_df=ind_df, **kwargs)
        # TODO: ds = ScaleDataset(ds)
        self.ds = Mult(ds, rate)

    def __len__(self): return len(self.ds)

    def __getitem__(self, *args, **kwargs):
        r = self.ds.__getitem__(*args, **kwargs)
        return r


class MainDatasetv2:
    def __init__(self,
                 cfg,
                 root,
                 ann_path,
                 ImgLoader,
                 AnnLoader,
                 organ,
                 data_source,
                 **kwargs):

        ds = ExtraValDataset(cfg, root, ann_path, ImgLoader, AnnLoader, organ, data_source, **kwargs)
        rate = kwargs.get('rate', 1)
        self.ds = Mult(ds, rate)

    def __len__(self): return len(self.ds)

    def __getitem__(self, *args, **kwargs):
        r = self.ds.__getitem__(*args, **kwargs)
        return r


class ExtraValDataset:
    def __init__(self,
                 cfg,
                 root,
                 ann_path,
                 ImgLoader,
                 AnnLoader,
                 organ,
                 data_source,
                 ext='png',
                 **kwargs):

        imgs = ImgLoader(root)
        anns = AnnLoader(ann_path)
        self.data = DataPair(imgs, anns)
        # rate = kwargs.pop('rate')
        imgs = list(root.glob(f'*.{ext}'))
        assert len(imgs) > 0, root

        label_kwargs = dict(lid=-1, organ=ORGANS[organ], data_source=DOMAINS[data_source], w=-1, h=-1, rle='')
        self.labels = [Label(fname=f.name, **label_kwargs) for f in imgs]

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        r = self.data(label)
        return r


def item_reader(item):
    i, a = item
    return i, a
