from collections import defaultdict

import torch
import metrics
from callbacks_fn import default_zero_tensor
from tools_tv import run_once, norm_2d, batch_quantile

from data import ORGANS
from tv import prepare_batch


import shallow as sh


class HubCB(sh.callbacks.Callback):
    def __init__(self, model_ema, batch_read=lambda x: x, batch_transform=sh.callbacks.batch_transform_cuda, logger=None):
        sh.utils.file_op.store_attr(self, locals())
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

    @sh.utils.call.on_mode(mode='VALID_HUB')
    def before_epoch(self):
        run_once.__wrapped__._clear()
        if self.cfg.PARALLEL.DDP: self.L.dl.sampler.set_epoch(self.L.n_epoch)

    @sh.utils.call.on_mode(mode='VALID_HUB')
    def after_epoch(self):
        collect_dist(self)

    @sh.utils.call.on_mode(mode='VALID_HUB')
    def step(self):
        if self.sched():
            self.run_hub()
        else:
            raise sh.exceptions.CancelEpochException

    def run_hub(self):
        ema = self.L.model_ema is not None
        model = self.L.model if not ema else self.L.model_ema.module
        batch = self.batch_read(self.L.batch)
        batch = prepare_batch(batch, self, train=False)

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                pred = model(batch)
                self.L.pred = pred

                pred_hm = pred['yb'].float()
                gt = batch['yb'].float()

                if pred_hm.shape[1] > 1: # multilabel mode
                    r = []
                    for i, hm in enumerate(pred_hm):
                        organ_idx = batch['cls'][i]
                        hm = hm[organ_idx:organ_idx+1]
                        r.append(hm)
                    pred_hm = torch.stack(r) # B,1,H,W

                all_organs_dice = metrics.calc_score(pred_hm, gt)
                op = 'gather'
                self.L.tracker_cb.set('dices_hub', all_organs_dice, operation=op)
                self.L.tracker_cb.set('classes_hub', batch['cls'].float(), operation=op)


def collect_dist(cb, ema=True, train=False):
    cb.L.tracker_cb._collect_all()
    dices = cb.L.tracker_cb.dices_hub.cpu()
    classes = cb.L.tracker_cb.classes_hub.cpu()

    dices = dices.view(-1, dices.shape[-1])
    classes = classes.view(-1, classes.shape[-1])
    ORGANS_DECODE = {v:k for k,v in ORGANS.items()}

    if cb.cfg.PARALLEL.IS_MASTER:
        # macro = []
        for i in range(5):
            idxs = classes.long() == i
            if not idxs.any(): continue
            class_name = ORGANS_DECODE[i]
            organ_dices = dices[idxs]
            organ_dice_mean = organ_dices.mean()
            organ_dice_std = organ_dices.std()
            cb.log_warning(f'\t {cb.L.mode} Dice {class_name:<20} mean {organ_dice_mean:<.3f}, std {organ_dice_std:<.3f} len {len(organ_dices)}')
            cb.L.writer.add_scalar(f'organs_{cb.L.mode}/{class_name}', organ_dice_mean, cb.L.n_epoch)
            # macro.append(organ_dice_mean)
        # cb.L.writer.add_scalar(f'organs/macro_avg', torch.as_tensor(macro).mean(), cb.L.n_epoch)
