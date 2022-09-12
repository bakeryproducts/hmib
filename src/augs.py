import random
from functools import partial

import cv2
import torch
import staintools
import torchstain
import numpy as np
import albumentations as albu
from albumentations.augmentations.functional import shift_rgb

import shallow as sh
from data import ORGANS, DOMAINS


class ColorAugs(albu.core.composition.OneOf):
    def __init__(self, *args, **kwargs):
        augs = [
            # albu.HueSaturationValue(128, 128, 0, p=1.0),
            albu.RGBShift(40, 40, 40, p=1.0),
            # albu.RandomBrightnessContrast(0.5, 0.5, p=1.0),
        ]
        super().__init__(augs, *args, **kwargs)


class NoiseAugs(albu.core.composition.OneOf):
    def __init__(self, *args, **kwargs):
        augs = [
            albu.MultiplicativeNoise((0.9, 1.1), per_channel=True, elementwise=True, p=0.5),
            albu.PixelDropout(dropout_prob=0.05, p=0.5),
            albu.ImageCompression(quality_lower=99, quality_upper=100, p=.5),
            #albu.Blur(p=1.0),
        ]
        super().__init__(augs, *args, **kwargs)


class DomainStainer(albu.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, domain='gtex', step=100, p=.5):
        super().__init__(p=p)
        self.domain = DOMAINS[domain]
        self.norms = {v:None for v in ORGANS.values()}
        self.fit_count = 0
        self.norm_count = 0
        self.update_step = step

    @property
    def targets(self):
        return {"image": self.apply}

    @property
    def targets_as_params(self):
        return ["organ", 'domain']

    def get_params_dependent_on_targets(self, params):
        return params

    def init_fit(self, dst, organ):
        try:
            n = torchstain.normalizers.MacenkoNormalizer(backend='torch')
            n.fit(torch.from_numpy(dst).permute(2,0,1))
        except RuntimeError as e:
            print('DOMAIN FIT RUNERROR!!', dst.shape, dst.max(), organ)
            print(e)
            n=None
        except IndexError as e:
            print('DOMAIN FIT INDERROR!!', dst.shape, dst.max(), dst.dtype, dst.min(), organ)
            print(e)
            n=None

        self.norms[organ] = n

    def sched(self, image, organ):
        # valid image stats?
        if self.norms[organ] is None:
            return True
        elif self.norm_count > self.update_step:
            self.norm_count = 0
            return True

    def apply(self, image, **params):
        organ = params['organ']
        domain = params['domain']

        if domain != self.domain:
            n = self.norms[organ]
            if n is not None:
                try:
                    image,_,_ = n.normalize(I=torch.from_numpy(image).permute(2,0,1), stains=False)
                    image = image.byte().numpy()
                except:
                    # can fail on specific pair?
                    self.norms[organ] = None
                    pass
                self.norm_count += 1

        elif self.sched(image, organ):
            self.init_fit(image, organ)
            self.fit_count += 1

        return image



class ColorMeanShift(albu.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, p=1.):
        super().__init__(p=p)
        shifts = {'largeintestine': [129, 116,138],
                  'lung': [198, 187, 207],
                  'spleen': [167, 92, 128],
                  'prostate': [187,140,194],
                  'kidney': [156,100,171],
                  }
        self.shifts = {ORGANS[k]:v for k,v in shifts.items()}

    @property
    def targets(self):
        return {"image": self.apply}

    @property
    def targets_as_params(self):
        return ["organ"]

    def get_params_dependent_on_targets(self, params):
        return params

    def apply(self, image, **params):
        organ = params['organ']
        tr,tg,tb = self.shifts[organ]
        r,g,b = image.mean((0,1))
        shift = tr-r, tg-g, tb-b
        image = shift_rgb(image, *shift)
        return image


class _ExampleAug(albu.core.transforms_interface.DualTransform):
    def __init__(self, p=1.):
        super().__init__(p=p)

    @property
    def targets(self):
        return {"image": self.apply, 'mask':self.apply}

    @property
    def targets_as_params(self):
        return ["seed"]

    def get_params_dependent_on_targets(self, params):
        return params

    def apply(self, image, **params):
        seed = params['seed'] # call should be like : albu.compose([aug1, aug2])(image=img, mask=m, seed=42)
        image = __my_fn_aug(image, seed)
        return image


def __my_fn_aug(image, seed):
    #logic
    return image




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
        self.aug = self._sup_aug

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, mask = item['x'], item['y']
        cls = item['cls']
        dom = item['dom']

        aug_images = []
        aug_masks = []
        mult = self.cfg.AUGS.AUG_MULTIPLICITY if self.train else 1

        for i in range(mult):
            aimage, amask = self.aug(image, mask, organ=cls, dom=dom)
            aug_images.append(aimage)
            aug_masks.append(amask)
            #print(aimage.shape, aimage.max(), aimage.dtype, amask.max(), mask.max())

        aitem = dict(x=aug_images, y=aug_masks)
        item['cls'] = mult*[cls]#np.hstack(mult*[cls])
        item.update(aitem)
        return item

    def _sup_aug(self, image, mask, organ, dom):
        kwargs = dict(image=image, mask=mask, organ=organ, domain=dom)
        res = self.transforms(**kwargs)
        return res['image'], res['mask']

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
