from collections import defaultdict

import torch
import torchvision.transforms as T

import metrics
from callbacks import _TrainCallback
from callbacks_fn import default_zero_tensor
from tools_tv import run_once, norm_2d, batch_quantile

from augment.mixup import MSR, MixUpAug
from augment.fmix import FMixAug, AmpAug
from augment.batch_augs import Splitter
from data import ORGANS


import shallow as sh



def prepare_batch(batch, cb, train):
    xb = batch['xb'].float()
    yb = batch['yb'].float()
    cls = batch['cls'].float()

    run_once(0, cb.log_debug, 'After load, XB', sh.utils.common.st(xb))
    run_once(1, cb.log_debug, 'After load, YB', sh.utils.common.st(yb))

    xb, yb = xb.cuda(), yb.cuda()
    #xb = batch_quantile(xb, q=.005)
    #run_once(2, cb.log_debug, 'quantiled, XB', sh.utils.common.st(xb))

    run_once(3, cb.log_debug, 'After train aug, XB', sh.utils.common.st(xb))
    run_once(35, cb.log_debug, 'After train aug, YB', sh.utils.common.st(yb))

    if cb.cfg.AUGS.NORM.MODE == 'meanstd':
        xb, mean, std = norm_2d(xb, mean=cb.cfg.AUGS.NORM.MEAN, std=cb.cfg.AUGS.NORM.STD)
    elif cb.cfg.AUGS.NORM.MODE == 'minmax':
        xmin = (xb.view(xb.shape[0], -1)).min(dim=1)[0]
        xmin = xmin.view(-1, 1, 1, 1)
        xmax = (xb.view(xb.shape[0], -1)).max(dim=1)[0]
        xmax = xmax.view(-1, 1, 1, 1)
        xb = (xb - xmin) / (xmax - xmin)
    elif cb.cfg.AUGS.NORM.MODE == 'const':
        xb = xb / 255.
    run_once(4, cb.log_debug, 'After 2d norm and aug, XB', sh.utils.common.st(xb))

    if train:
        xb = cb.colj(xb)
        xb = cb.gaub(xb)

    b,_,h,w = xb.shape
    cls_layer = torch.ones(b,1,h,w).float()
    cls_layer = cls_layer * cls.view(-1, 1,1,1)
    cls_layer = (cls_layer.to(xb) + 1) / 5
    xb = torch.hstack([xb, cls_layer])
    run_once(9, cb.log_debug, 'cls layer xb', sh.utils.common.st(xb))

    if train:
        xb, yb = cb.mixup(xb, yb)
        xb, yb = cb.fmix(xb, yb)
        xb, yb = cb.msr(xb, yb, cb)

    if cb.clamp is not None: xb.clamp_(-cb.clamp,cb.clamp)


    batch['xb'] = xb
    batch['yb'] = yb
    cls = batch['cls'].cuda().flatten()
    return dict(xb=xb, yb=yb, cls=cls, mask=None)



class TrainCB(_TrainCallback):
    def __init__(self, batch_read=lambda x: x, amp_scaler=None, logger=None):
        sh.utils.file_op.store_attr(self, locals())

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']
        self.L.amp_scaler = self.amp_scaler
        self.clamp = self.cfg.FEATURES.CLAMP
        self.grad_clip = self.cfg.FEATURES.GRAD_CLIP
        self.grad_clip_mode = self.cfg.FEATURES.CLIP_MODE
        self.sam = self.cfg.FEATURES.SAM.RHO > 0
        self.msr = MSR(self.cfg)
        self.mixup = MixUpAug(self.cfg)
        self.fmix = FMixAug(self.cfg)
        # self.noise = NoiseInjection(max_noise_level=.15, p=.2)
        # self.ampaug = AmpAug(scale=20, p=.2)
        self.gaub = Splitter(aug=T.GaussianBlur(kernel_size=3, sigma=7), p=.3).cuda()
        self.colj = Splitter(aug=T.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.4), p=.7).cuda()

        self.batch_acc_step = self.cfg.FEATURES.BATCH_ACCUMULATION_STEP
        self.loss_weights = {l.name:float(l.weight) for l in self.cfg.LOSS}
        self.loss_weights['ssl'] = 1


    @sh.utils.call.on_mode(mode='TRAIN')
    def before_epoch(self):
        run_once.__wrapped__._clear()
        try:
            if self.cfg.PARALLEL.DDP: self.L.dl.sampler.set_epoch(self.L.n_epoch)
        except AttributeError as e:
            pass

    def sched(self): return (self.L.n_epoch % 5) == 0

    @sh.utils.call.on_mode(mode='TRAIN')
    def step(self):
        self.supervised_train_step()
        if self.cfg.TRAIN.EMA.ENABLE: self.L.model_ema.update(self.L.model, self.L.ema_decay)

    def sam_reduce(self, batch, p=.2):
        new_l = int(batch['xb'].shape[0] * p)
        b = {}
        for k,v in batch.items():
            b[k] = v[:new_l]
        return b

    def supervised_train_step(self):
        with torch.cuda.amp.autocast(enabled=self.amp_scaler.is_enabled()):
            batch = self.batch_read(self.L.batch)
            batch_second = prepare_batch(batch, self, train=True)
            batch = self.sam_reduce(batch_second, p=self.cfg.FEATURES.SAM.REDUCE) if self.sam else batch_second

            loss_d = defaultdict(default_zero_tensor)

            pred = self.L.model(batch)
            self.L.pred = pred
            run_once(41, self.log_debug, 'Pred', sh.utils.common.st(pred['yb']))

            loss_d.update(self.L.loss_func(pred, batch))
            total_loss = self.get_total_loss(loss_d, tracking=True, ohem=None)

        self.update_step(total_loss)
        self.sam_update(batch_second['xb'], batch_second) # will check sam flag from cfg


class ValCB(sh.callbacks.Callback):
    def __init__(self, model_ema, batch_read=lambda x: x, batch_transform=sh.callbacks.batch_transform_cuda, logger=None):
        sh.utils.file_op.store_attr(self, locals())
        self.evals = []
        self.once_flag = False

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']
        self.L.model_ema = self.model_ema
        self.loss_kwargs = {}
        self.clamp = self.cfg.FEATURES.CLAMP

    def sched(self):
        e = self.L.n_epoch
        if e <= self.cfg.TRAIN.START_VAL: return False
        elif e % self.cfg.TRAIN.SCALAR_STEP == 0:
            return True

    @sh.utils.call.on_mode(mode='VALID')
    def before_epoch(self):
        run_once.__wrapped__._clear()
        if self.cfg.PARALLEL.DDP: self.L.dl.sampler.set_epoch(self.L.n_epoch)

    @sh.utils.call.on_mode(mode='VALID')
    def after_epoch(self):
        collect_map_score(self)

    @sh.utils.call.on_mode(mode='VALID')
    def step(self):
        if self.sched() and self.cfg.MODEL.ARCH != 'ssl':
            self.run_valid()
        else:
            raise sh.exceptions.CancelEpochException


    def run_valid(self):
        ema = self.L.model_ema is not None
        model = self.L.model if not ema else self.L.model_ema.module
        model.eval()
        prefix = 'val' if not ema else 'ema'
        batch = self.batch_read(self.L.batch)
        batch = prepare_batch(batch, self, train=False)

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                pred = model(batch)
                self.L.pred = pred

                pred_hm = pred['yb'].float()
                gt = batch['yb'].float()
                if pred_hm.shape[1] > 1: # multilabel mode
                    pred_single = []
                    gt_single= []
                    for i, hm in enumerate(pred_hm):
                        organ_idx = batch['cls'][i]
                        hm = hm[organ_idx:organ_idx+1]
                        pred_single.append(hm)

                        gto = gt[i, organ_idx:organ_idx+1]
                        gt_single.append(gto)

                    pred_hm = torch.stack(pred_single) # B,1,H,W
                    gt = torch.stack(gt_single) # B,1,H,W

                #gt = gt.sum(1, keepdims=True)
                # print(gt.shape, pred_hm.shape)
                all_organs_dice = metrics.calc_score(pred_hm, gt)

                is_lung = torch.tensor([i == ORGANS['lung'] for i in batch['cls']])
                pred_hm[is_lung] = gt[is_lung]
                dice_fix_lung = metrics.calc_score(pred_hm, gt)

                op = 'gather'
                self.L.tracker_cb.set('dices', all_organs_dice, operation=op)
                self.L.tracker_cb.set('classes', batch['cls'].float(), operation=op)

                # pred_cls = pred['cls'].softmax(1)
                # pred_cls = torch.max(pred_cls, 1)[1]
                # gt_cls = batch['cls']
                # acc = (pred_cls == gt_cls).float().mean()
                # self.L.tracker_cb.set('cls_acc', acc)

                self.L.tracker_cb.set('ema_score', dice_fix_lung)
                self.L.tracker_cb.set('score', dice_fix_lung)

                loss_d = defaultdict(default_zero_tensor)
                loss_d.update(self.L.loss_func(pred, batch, **self.loss_kwargs))

                for k, loss in loss_d.items():
                    if loss is None: continue
                    loss = loss.mean()
                    if loss.isnan(): loss = torch.zeros(1).cuda()
                    self.L.tracker_cb.set(f'{k}_{prefix}_loss', loss)


def collect_map_score(cb, ema=True, train=False):
    cb.L.tracker_cb._collect_all()
    dices = cb.L.tracker_cb.dices.cpu()
    classes = cb.L.tracker_cb.classes.cpu()

    dices = dices.view(-1, dices.shape[-1])
    classes = classes.view(-1, classes.shape[-1])
    ORGANS_DECODE = {v:k for k,v in ORGANS.items()}
    LB_WEIGHT = {
        "kidney": 0.176, 
        "prostate": 0.219, 
        "spleen": 0.252, 
        "largeintestine": 0.096, 
        "lung": 0.257,
    }

    if cb.cfg.PARALLEL.IS_MASTER:
        prefix = f'organs_{cb.L.mode}'
        macro = []
        lb_avg = 0
        for i in range(5):
            idxs = classes.long() == i
            class_name = ORGANS_DECODE[i]
            organ_dices = dices[idxs]
            organ_dice_mean = organ_dices.mean()
            organ_dice_std = organ_dices.std()
            cb.log_warning(f'\t {cb.L.mode} Dice {class_name:<20} mean {organ_dice_mean:<.3f}, std {organ_dice_std:<.3f} len {len(organ_dices)}')
            cb.L.writer.add_scalar(f'{prefix}/{class_name}', organ_dice_mean, cb.L.n_epoch)
            macro.append(organ_dice_mean)
            lb_avg += LB_WEIGHT[class_name] * organ_dice_mean
        cb.L.writer.add_scalar(f'{prefix}/macro_avg', torch.as_tensor(macro).mean(), cb.L.n_epoch)
        cb.L.writer.add_scalar(f'{prefix}/lb_avg', torch.as_tensor(lb_avg), cb.L.n_epoch)

