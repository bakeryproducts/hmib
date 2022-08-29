import importlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach
from omegaconf import OmegaConf

from data import ORGANS
from tools_tv import batch_quantile


def norm_2d(xb, mode='batch', mean=None, std=None):
    if mean is None:
        mean, std = xb.mean(), xb.std()
    else:
        mean = torch.tensor(mean).view(1,1,1,1).to(xb)
        std = torch.tensor(std).view(1,1,1,1).to(xb)

    if mode == 'batch':
        xb = (xb - mean) / std
    elif mode == 'channel':
        xb = (xb - xb.mean((0,2,3), keepdims=True)) / xb.std((0,2,3), keepdims=True)
    elif mode == 'example':
        xb = (xb - xb.mean((1,2,3), keepdims=True)) / xb.std((1,2,3), keepdims=True)
    elif mode == 'chexamp':
        xb = (xb - xb.mean((2,3), keepdims=True)) / xb.std((2,3), keepdims=True)
    elif mode == 'freq_norm':
        xb = (xb - xb.mean((0,1,3), keepdims=True)) / xb.std((0,1,3), keepdims=True)
    else:
        raise Exception
    return xb, mean, std


@torch.no_grad()
def model_run(xb, model, preprocess, act=True):
    with torch.cuda.amp.autocast():
        xb = preprocess(xb)
        r = model(xb)
        if act:
            r['yb'] = r['yb'].softmax(1)
            r['posb'] = r['posb'].sigmoid()
            r['keyf'] = r['keyf'].sigmoid()
        rr = {}
        for k,v in r.items():
            rr[k] = v.cpu()
    return r


def preprocess(xb):
    #xb = batch_quantile(xb)
    xb,_,_ = norm_2d(xb, mode='example')
    xb.clamp_(-10,10)
    return xb


class RawBatchModel(torch.nn.Module):
    """
    Model wrapper that convert input and output format
    from dictionary to raw tensors.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, batch):
        return self.model({"xb": batch})["yb"]


class Inferer:
    def __init__(
        self,
        model,
        cfg,
        threshold=0.5,
        sigmoid=True,
        tta=False,
        tta_merge_mode="mean",
        to_gpu=False,
    ):
        """
        Params
        ------
        model: torch.nn.Module
            Model
        cfg: OmegaConf.Config
            Config
        threshold: float, default 0.5
            Confidence threshold
            If None no thresholding will be done
        sigmoid: bool, default True
            Apply sigmoid or not
        tta: bool, default False
            Apply TTA or not
        tta_merge_mode: str, default "mean"
            TTA merge mode, should be one of [mean, max]
        """
        self.model = RawBatchModel(model)
        self.cfg = cfg

        self.threshold = threshold
        self.sigmoid = sigmoid
        self.tta = tta
        self.tta_merge_mode = tta_merge_mode
        self.to_gpu = to_gpu

        if self.tta:
            self.model = ttach.SegmentationTTAWrapper(
                self.model,
                ttach.aliases.d4_transform(),
                merge_mode=self.tta_merge_mode
            )

    def preprocess(self, batch):
        """Preprocessing

        Params
        ------
        batch: np.array of shape (batch, height, width, channels); batch of raw images.

        Returns
        -------
        X: torch.Tensor of shape (batch, channels, height, width); batch of preprocessed images.
        """
        X = torch.from_numpy(batch).float()
        if self.to_gpu:
            X = X.cuda()
        X = batch_quantile(X, q=.005)
        X = X.permute((0, 3, 1, 2))
        X, mean, std = norm_2d(X, mean=self.cfg.AUGS.MEAN, std=self.cfg.AUGS.STD)
        X.clamp_(-self.cfg.FEATURES.CLAMP, self.cfg.FEATURES.CLAMP)
        return X

    def __call__(self, batch, organ=None):
        """Inference

        Params
        ------
        batch: torch.Tensor of shape (batch, channels, height, width)
            Batch of preprocessed images
        organ: str, optional, default None
            Which organ is on the images
            None means we have no such information,
            in this case max probability among all organs will be used

        Returns
        -------
        yb: np.array of shape (batch, height, width, 1)
            Batch masks
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                batch_pred = self.model(self.preprocess(batch))

        yb = batch_pred.float()

        if self.sigmoid:
            yb.sigmoid_()

        yb = yb.cpu().numpy()
        yb = yb.transpose((0, 2, 3, 1))

        # Extract organ mask
        if organ is not None:
            yb = yb[..., ORGANS[organ]][..., None]

        # Extract mask for all organs
        else:
            yb = torch.max(yb, dim=-1, keepdim=True)

        # Binarize the mask
        if self.threshold is not None:
            yb = (yb > self.threshold).astype(np.uint8)

        return yb

    @classmethod
    def create(cls, model_file, config_file, experiment_dir, to_gpu=False, **kwargs):
        # Load config
        cfg = OmegaConf.load(config_file)
        cfg.MODEL.ENCODER.pretrained = False

        network_module = init_modules(experiment_dir, "network")

        # Load model
        model = init_model(cfg, model_file, network_module, to_gpu)

        return cls(model, cfg, to_gpu=to_gpu, **kwargs)


class EnsembleInfer:
    def __init__(self, infers):
        self.infers = infers
        self.mode = 'avg'

    def __call__(self, xb, **kwargs):
        res = defaultdict(list)
        for inf in self.infers:
            pred = inf(xb, **kwargs)
            for k,v in pred.items():
                res[k].append(v)
            #res.append(pred)
        reduced = {}
        for k,v in res.items():
            v = torch.stack(v)
            if self.mode == 'avg':v = v.mean(0)
            elif self.mode == 'max':v = v.max(0)
            reduced[k] = v

        return reduced


def get_model(model_path, fix_cfg=lambda x:x, **kwargs):
    model_root = model_path.parent.parent.parent
    cfg_path = model_root / 'src/configs/xstage1.yaml'
    cfg = OmegaConf.load(cfg_path)
    cfg = fix_cfg(cfg)
    # cfg.MODEL.ENCODER.pretrained = False
    network = init_modules(model_root)
    model = init_model(cfg, model_path, network, **kwargs)
    return model


def init_data_module_from_checkpoint(root, name, file_name):
    spec = importlib.util.spec_from_file_location(name, str(root/f'src/{file_name}'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def init_modules(p, module_name='network'):
    p = Path(p)
    sys.path.insert(0, str(p / 'src'))

    if module_name in sys.modules:
        del sys.modules[module_name]

    module = init_data_module_from_checkpoint(p, module_name, f'{module_name}.py')
    sys.path.pop(0)
    return module


def init_model(cfg, model_path, network, to_gpu=False):
    # root = model_path.parent.parent.parent
    model = network.model_select(cfg)()
    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model['model_state']['cls'])
    model = model.eval()
    if to_gpu:
        model = model.cuda()
    return model


class CTTA(torch.nn.Module):
    def __init__(self, infer, transformers, keys_demask=[], keys_1d=[], mode='avg'):
        super().__init__()
        self.transformers = transformers
        self.infer = infer
        self.keys_demask = keys_demask
        self.keys_1d = keys_1d
        self.mode = mode

    def forward(self, xb, **kwargs):
        res = defaultdict(list)
        for transformer in self.transformers: # custom transforms or e.g. tta.aliases.d4_transform()
            axb = transformer.augment_image(xb)
            #print(transformer, axb.shape)
            pred = self.infer(axb.contiguous(), **kwargs)
            assert isinstance(pred, dict), type(pred)
            for k,v in pred.items():
                if k in self.keys_demask:
                    v = transformer.deaugment_mask(v)
                elif k not in self.keys_1d:
                    continue
                res[k].append(v)

        for k,v in res.items():
            if self.mode == 'avg':
                res[k] = torch.stack(v).mean(0)
            elif self.mode == 'max':
                res[k] = torch.stack(v).max(0)[0]

        return res


def get_tta(infer, transformers):
    tta_infer = CTTA(infer,
                     transformers=transformers,
                     keys_demask=['yb'],
                     keys_1d=['posb', 'keyf'])
    return tta_infer


def scale(x, lp=5, rp=95):
    l,r = np.percentile(x, lp), np.percentile(x, rp)
    xc = np.clip(x, l,r)
    x = (xc - xc.min()) / (xc.max()-xc.min())
    return x


def cut(i):
    scaled = scale(i, 5, 95)
    binar = scaled > .3

    output = cv2.connectedComponentsWithStats((binar*255).astype(np.uint8), 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    ii = np.argsort(stats[:,-1])[::-1]
    idx = ii[1]

    x,y,w,h,_ = stats[idx]
    cutted = i[y:y+h, x:x+w]
    return cutted, y,x,h,w


def infer_loader(p):
    SIZE = (512, 512)

    i = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    H,W = i.shape
    i, y,x,h,w = cut(i) # space cut
    i = cv2.resize(i, SIZE)
    i = torch.from_numpy(i.astype(np.float32))

    rq = torch.quantile(i, .995)
    lq = torch.quantile(i, .005)
    iq = torch.clamp(i, lq, rq)
    iq = (iq - iq.min()) / (iq.max() - iq.min())
    return i, np.array((H,W,y,x,h,w))


def rescale_mask(mask, params):
    H,W,y,x,h,w = params.flatten().int().numpy()
    #print(H,W,y,x,h,w)
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
    buf = np.zeros((H,W,3), dtype=np.float32)
    buf[y:y+h, x:x+w] = mask
    return buf
