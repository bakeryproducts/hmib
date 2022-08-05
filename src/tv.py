import random
from collections import defaultdict

import torch
import torchvision.transforms as T

import metrics
from callbacks import _TrainCallback
from callbacks_fn import default_zero_tensor
from tools_tv import run_once, norm_2d, batch_quantile

from mixup import MSR, MixUpAug
from fmix import FMixAug, AmpAug
from data import MaskGenerator, ORGANS


import shallow as sh



def prepare_batch(batch, cb, train):
    xb = batch['xb'].float()
    yb = batch['yb'].float()

    run_once(0, cb.log_debug, 'After load, XB', sh.utils.common.st(xb))
    run_once(1, cb.log_debug, 'After load, YB', sh.utils.common.st(yb))

    xb, yb = xb.cuda(), yb.cuda()
    # yb = yb / 255.

    xb = batch_quantile(xb, q=.005)
    run_once(2, cb.log_debug, 'quantiled, XB', sh.utils.common.st(xb))

    if train:
        # cc = xb.byte().chunk(16)
        # xb = torch.vstack([cb.augmenter(c) for c in cc])
        # xb = xb.float()
        # xb, yb = cb.mixup(xb, yb)
        xb, yb = cb.fmix(xb, yb)
        xb, yb = cb.msr(xb, yb, cb)
    #     xb = cb.ampaug(xb)
    #     xb, yb = cb.hdr(xb, yb)

    run_once(3, cb.log_debug, 'After train aug, XB', sh.utils.common.st(xb))
    run_once(35, cb.log_debug, 'After train aug, YB', sh.utils.common.st(yb))

    xb, mean, std = norm_2d(xb, mean=cb.cfg.AUGS.MEAN, std=cb.cfg.AUGS.STD)
    # xb = (xb-xb.min()) / (xb.max() - xb.min())
    # xb = xb / 255.
    run_once(4, cb.log_debug, 'After 2d norm and aug, XB', sh.utils.common.st(xb))

    # if train:
    #     xb = cb.noise(xb)

    if cb.clamp is not None: xb.clamp_(-cb.clamp,cb.clamp)

    # mask = []
    # progress = cb.L.np_epoch
    # run_once(44, cb.log_debug, 'Progress ', progress)
    # for i in range(xb.shape[0]):
    #     m = cb.mg(progress)
    #     m = torch.from_numpy(m)
    #     mask.append(m)
    # mask = torch.stack(mask,0)
    # mask = mask.cuda()
    mask = None


    batch['xb'] = xb
    batch['yb'] = yb
    batch['mask'] = mask
    cls = batch['cls'].cuda().flatten()
    return dict(xb=xb, yb=yb, cls=cls, mask=mask)



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
        # self.hdr = HDRAug(max_shift=3, p=.1)
        # self.ampaug = AmpAug(scale=20, p=.2)
        self.augmenter = T.TrivialAugmentWide()
        self.mg = MaskGenerator(input_size=192, mask_ratio=.3, model_patch_size=4)

        self.batch_acc_step = self.cfg.FEATURES.BATCH_ACCUMULATION_STEP

        self.loss_weights = {l.name:float(l.weight) for l in self.cfg.LOSS}
        self.loss_weights['ssl'] = 1


    @sh.utils.call.on_train
    def before_epoch(self):
        run_once.__wrapped__._clear()
        try:
            if self.cfg.PARALLEL.DDP: self.L.dl.sampler.set_epoch(self.L.n_epoch)
        except AttributeError as e:
            pass

    def sched(self): return (self.L.n_epoch % 5) == 0

    def train_step(self):
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

            # pred_cls = pred['cls'].softmax(1)
            # pred_cls = torch.max(pred_cls, 1)[1]
            # gt_cls = batch['cls']
            # acc = (pred_cls == gt_cls).float().mean()
            # self.L.tracker_cb.set('cls_acc', acc)

            # sega = pred['yb']
            # sega = sega.sigmoid()
            # sega = (sega > .5) .float()

            # segb = batch['yb']
            # # print(sega.shape, segb.shape, )
            # dice = metrics.calc_score(sega, segb)
            # run_once(441, self.log_warning, 'dice', dice)

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
        self.mg = MaskGenerator(input_size=192, mask_ratio=.5, model_patch_size=4)

    def sched(self):
        e = self.L.n_epoch
        if e <= self.cfg.TRAIN.START_VAL: return False
        elif e % self.cfg.TRAIN.SCALAR_STEP == 0:
            return True

    @sh.utils.call.on_validation
    def before_epoch(self):
        run_once.__wrapped__._clear()
        # if self.cfg.PARALLEL.DDP: self.L.dl.sampler.set_epoch(self.L.n_epoch)

    @sh.utils.call.on_validation
    def after_epoch(self):
        collect_map_score(self)

    @sh.utils.call.on_validation
    def val_step(self):
        if self.sched() and self.cfg.MODEL.ARCH != 'ssl':
            self.run_valid()
        else:
            raise sh.learner.CancelEpochException

    def run_valid(self):
        ema = self.L.model_ema is not None
        model = self.L.model if not ema else self.L.model_ema.module
        prefix = 'val' if not ema else 'ema'
        batch = self.batch_read(self.L.batch)
        batch = prepare_batch(batch, self, train=False)

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                pred = model(batch)
                self.L.pred = pred

                if self.cfg.MODEL.ARCH != 'ssl':
                    sega = pred['yb']
                    sega = sega.sigmoid()

                    segb = batch['yb']
                    dice = metrics.calc_score(sega, segb)

                    # run_once(441, self.log_warning, 'dice', sh.utils.common.st(dice))
                    # run_once(442, self.log_warning, 'classes', sh.utils.common.st(batch['cls'].float()))

                    op = 'gather'
                    self.L.tracker_cb.set('dices', dice.unsqueeze(0), operation=op)
                    self.L.tracker_cb.set('classes', batch['cls'].float().unsqueeze(0), operation=op)

                    pred_cls = pred['cls'].softmax(1)
                    pred_cls = torch.max(pred_cls, 1)[1]
                    gt_cls = batch['cls']
                    acc = (pred_cls == gt_cls).float().mean()
                    self.L.tracker_cb.set('cls_acc', acc)

                else:
                    dice = torch.zeros(1).cuda()


                self.L.tracker_cb.set('ema_dice', dice)
                # self.L.tracker_cb.set('val_score', dice)
                self.L.tracker_cb.set('ema_score', dice)
                self.L.tracker_cb.set('score', dice)

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

    if cb.cfg.PARALLEL.IS_MASTER:
        for i in range(5):
            idxs = classes.long() == i
            class_name = ORGANS_DECODE[i]
            organ_dices = dices[idxs]
            organ_dice_mean = organ_dices.mean()
            organ_dice_std = organ_dices.std()
            cb.log_warning(f'\t Dice {class_name:<20} mean {organ_dice_mean:<.3f}, std {organ_dice_std:<.3f} len {len(organ_dices)}')
            cb.L.writer.add_scalar(f'organs/{class_name}', organ_dice_mean, cb.L.n_epoch)
