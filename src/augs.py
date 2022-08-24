import random
from functools import partial

import cv2
import torch
import staintools
import numpy as np
import albumentations as albu

import shallow as sh


class ShiftScaleRotate(albu.ShiftScaleRotate):
    def get_params(self):
        params = super().get_params()

        # Make scale to be < 0 and > 0 with equal probabilities
        if self.scale_limit[0] * self.scale_limit[1] < 0:
            if np.random.random() < 0.5:
                params["scale"] = np.random.uniform(self.scale_limit[0], 0)
            else:
                params["scale"] = np.random.uniform(0, self.scale_limit[1])

        return params


def pixel_scale(image, scale_factor):
    result = staintools.LuminosityStandardizer.standardize(image)

    result = cv2.resize(result, dsize=None, fx=scale_factor, fy=scale_factor)
    result = cv2.resize(result, dsize=(image.shape[1], image.shape[0]))

    result = staintools.LuminosityStandardizer.standardize(result)

    return result


class PixelScale(albu.ImageOnlyTransform):
    def __init__(self, scale_limit=[0.8, 12.], always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.scale_limit = scale_limit

    def apply(self, image, scale, **params):
        return pixel_scale(image, scale)

    def get_params(self):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def get_transform_init_args_names(self):
        return ("scale_limit",)


class AugDataset:
    def __init__(self, cfg, dataset, transforms, train, ):
        self.cfg = cfg
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
        self.train = train
        self.aug = self._ssl_aug if cfg.MODEL.ARCH == 'ssl' else self._sup_aug

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, mask = item['x'], item['y']

        aug_images = []
        aug_masks = []
        mult = self.cfg.AUGS.AUG_MULTIPLICITY if self.train else 1

        for i in range(mult):
            aimage, amask = self.aug(image, mask, seed=0)
            aug_images.append(aimage)
            aug_masks.append(amask)
            # print(aimage.shape, aimage.max(), aimage.dtype)

        aitem = dict(x=aug_images, y=aug_masks)
        cls = item['cls']
        item['cls'] = np.hstack(mult*[cls])
        item.update(aitem)
        return item

    def _sup_aug(self, image, mask, seed):
        kwargs = dict(image=image, mask=mask, seed=seed)
        res = self.transforms(**kwargs)
        return res['image'], res['mask']

    def _ssl_aug(self, image, mask, seed):
        kwargs = dict(image=image, seed=seed)
        res = self.transforms(**kwargs)
        return res['image'], mask

    def __len__(self): return len(self.dataset)


class Augmentator(sh.augmentation.AugmentatorBase):
    def __init__(self, *args, **kwargs):
        super(Augmentator, self).__init__(*args, **kwargs)

    def aug_train(self): return self.compose(self._train())
    def aug_train_hard(self): return self.compose(self._train_hard())
    def aug_valid(self): return self.compose(self._valid())
    def aug_test(self): return self.compose(self._test())

    def _train(self):
        augs = [
            sh.augmentation.ToTensor(),
        ]
        return augs

    def _valid(self):
        augs = [
            sh.augmentation.ToTensor(),
        ]
        return augs


def maybe_eval(name, evaler=eval):
    return evaler(name) if isinstance(name, str) else name


def parse_augd(augd, evaler=maybe_eval):
    m = evaler(augd['m'])
    args = [evaler(i) for i in augd.get('args', [])]
    kwargs = {}
    for k,v in augd.get('kwargs', {}).items():
        kwargs[k] = evaler(v)

    return m(*args, **kwargs)


def parse_cfg_augs(aug_cfg):
    augs = []
    for augd in aug_cfg:
        aug = parse_augd(augd)
        augs.append(aug)
    return augs


def augment_dataset(cfg, k, ds, compose, **kwargs):
    augs = parse_cfg_augs(cfg.AUGS.get(k).AUGS)
    transforms = compose(augs)
    train = k == 'TRAIN'
    ad = AugDataset(cfg, ds, transforms, train, **kwargs)
    return ad


def create_augmented(cfg, dss, **kwargs):
    compose = albu.Compose
    aug_dss = {}
    for k, ds in dss.items():
        ads = augment_dataset(cfg, k, ds, compose, **kwargs)
        aug_dss[k] = ads
    return aug_dss
