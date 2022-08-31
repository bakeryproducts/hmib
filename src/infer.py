import importlib
import sys
from collections import defaultdict
from pathlib import Path

import torch
import ttach
from omegaconf import OmegaConf


def batch_quantile(b, q=.01):
    br = b.view(b.shape[0],-1)
    rq = torch.quantile(br, dim=1, q=1-q).view(-1,1,1,1)
    lq = torch.quantile(br, dim=1, q=  q).view(-1,1,1,1)
    return torch.max(torch.min(b, rq), lq)


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
        self.device = model.device

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
        X = batch.float().to(self.device)
        #X = batch_quantile(X, q=.005) # bchw

        if self.cfg.AUGS.NORM.MODE == 'meanstd':
            X, _, _ = norm_2d(X, mean=self.cfg.AUGS.NORM.MEAN, std=self.cfg.AUGS.NORM.STD)
        elif self.cfg.AUGS.NORM.MODE == 'minmax':
            X = (X - X.min()) / (X.max() - X.min())

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
        return yb

    @classmethod
    def create(cls, model_file, config_file, experiment_dir, device="cpu", **kwargs):
        # Load config
        cfg = OmegaConf.load(config_file)
        cfg.MODEL.ENCODER.pretrained = False

        network_module = init_modules(experiment_dir, "network")

        # Load model
        model = init_model(cfg, model_file, network_module, device)

        return cls(model, cfg, **kwargs)


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
        # TODO del submodules

    module = init_data_module_from_checkpoint(p, module_name, f'{module_name}.py')
    sys.path.pop(0)
    return module


def init_model(cfg, model_path, network, device="cpu"):
    model = network.model_select(cfg)()
    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model['model_state']['cls'])
    model = model.to(device).eval()
    model.device = device
    return model
