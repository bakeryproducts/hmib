from collections import defaultdict

import torch
import torchvision
import numpy as np
from timm.utils.agc import adaptive_clip_grad

from tools_tv import run_once
from callbacks_fn import weight_watcher, set_dropout, freeze_stages
import shallow as sh


def default_zero_tensor(): return lambda: torch.tensor([0.])


def get_ema_decay_cb(cfg):
    # lrcb = sh.callbacks.ParamSchedulerCB('before_epoch', 'lr', sched)
    all_scheds = dict(linear=sh.schedulers.sched_lin,
                      cos=sh.schedulers.sched_cos,
                      const=sh.schedulers.sched_const,
                      exp=sh.schedulers.sched_exp,)

    sched = all_scheds[cfg.TRAIN.EMA.type](cfg.TRAIN.EMA.start, cfg.TRAIN.EMA.end)
    emacb = sh.callbacks.ParamSchedulerCB('before_epoch', 'ema_decay', sched)
    return emacb


class TrackerItem:
    def __init__(self, value, operation='reduce', **kwargs): sh.utils.file_op.store_attr(self, locals())
    def __repr__(self): return f'{self.value}'


class DropoutCB(sh.callbacks.Callback):
    def __init__(self, logger=None):
        sh.utils.file_op.store_attr(self, locals())

    def before_epoch(self):
        drop_rate = self.sched()
        set_dropout(self.L.model.module, drop_rate)

    def sched(self):
        raise NotImplementedError


class WeightWatchCB(sh.callbacks.Callback):
    def __init__(self, logger=None):
        sh.utils.file_op.store_attr(self, locals())

    @sh.utils.call.on_train
    def before_epoch(self):
        weight_watcher(self)


class ScorerCB(sh.callbacks.Callback):
    def __init__(self, logger=None):
        sh.utils.file_op.store_attr(self, locals())

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']
        self.max_EMA_score = -1
        self.max_VAL_score = -1
        self.chpt_cb = sh.utils.call.get_cb_by_instance(self.L.cbs, CheckpointCB)


    def get_score(self, ema=True):
        return self.max_EMA_score

    def update_max_and_save(self, score, prefix, ema=False):
        suffix = 'ema' if ema else 'val'
        score = score.mean().item()
        if ema:
            if score > self.max_EMA_score:
                self.max_EMA_score = score
                if self.chpt_cb is not None:
                    self.chpt_cb.do_saving(f'{prefix}_{suffix}_{round(score, 4)}', save_ema=ema)
        else:
            if score > self.max_VAL_score:
                self.max_VAL_score = score
                if self.chpt_cb is not None:
                    self.chpt_cb.do_saving(f'{prefix}_{suffix}_{round(score, 4)}', save_ema=ema)

    @sh.utils.call.on_validation
    def after_epoch(self):
        if self.cfg.PARALLEL.IS_MASTER:
            if hasattr(self.L.tracker_cb, 'val_score'):
                self.update_max_and_save(self.L.tracker_cb.val_score, prefix='cmax', ema=False)
            if hasattr(self.L.tracker_cb, 'ema_score'):
                if self.L.n_epoch % self.cfg.TRAIN.SAVE_STEP == 0:
                    suffix = 'ema'
                    score = self.L.tracker_cb.ema_score
                    score = score.mean().item()
                    if score > self.max_EMA_score:
                        self.max_EMA_score = score
                    if self.chpt_cb is not None:
                        self.chpt_cb.do_saving(f'step_{suffix}_{round(score, 4)}', save_ema=suffix=='ema')
                elif self.cfg.TRAIN.EMA.ENABLE:
                    self.update_max_and_save(self.L.tracker_cb.ema_score, prefix='cmax', ema=True)


class TrackResultsCB(sh.callbacks.Callback):
    def __init__(self, logger=None):
        sh.utils.file_op.store_attr(self, locals())
        self.already_collected = False

    def _gather_list(self, li):
        if self.cfg.PARALLEL.DDP: li = sh.utils.distributed.gather_tensor(li)
        return li

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']
        self.accs = defaultdict(list)

    def set(self, name, value, **kwargs):
        item = TrackerItem(value, **kwargs)
        self.accs[name].append(item)

    def reset(self): self.accs = defaultdict(list)

    def before_epoch(self):
        self.reset()
        self.already_collected = False

    def after_epoch(self):
        if not self.already_collected: self._collect_all()

    def _collect_all(self):
        for k, track_item in self.accs.items():
            # print(k, track_item[:3])
            value = [i.value for i in track_item]
            # print(k)
            # print([i.value.shape for i in track_item])
            value = torch.vstack(value)
            value = value.unsqueeze(0)
            # print(k, value.shape, value.dtype)

            if track_item[0].operation == 'reduce':
                if self.cfg.PARALLEL.DDP: value = sh.utils.distributed.reduce_tensor(value)
                value = value.mean()
                setattr(self, k, value)
            if track_item[0].operation == 'gather':
                # print(k, value.shape, value.dtype)
                value = self._gather_list(value)
                setattr(self, k, value)
            if track_item[0].operation == 'none':
                # print(k, value.shape, value.dtype)
                setattr(self, k, value)
            # print(k, value.shape, value.dtype)

        self.lr = self.L.lr
        self.already_collected = True


class GradLogger(sh.callbacks.Callback):
    def __init__(self, batch_read=lambda x: x, amp_scaler=None, logger=None):
        sh.utils.file_op.store_attr(self, locals())

    def _grad_norm(self):
        shared_device = self.L.opt.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.L.opt.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def _log_grads(self, m):
        for tag, param in m.named_parameters():
            if param.grad is not None:
                pass
                #self.log_debug(tag, param.max(), param.min())
                #self.log_debug(tag, param.grad.max(), param.grad.min())


class EarlyExit(sh.callbacks.Callback):
    # TODO redo this, bump from val cb? val each epoch for now
    def __init__(self, score_cb, wait_epoch=5, start_p_epoch=0., logger=None):
        sh.utils.file_op.store_attr(self, locals())
        self.current_streak = 0
        self.max_score = -1
        self.exit_signal = torch.tensor([0]).cuda()
        self.start_p_epoch = start_p_epoch
        #print(f'EARLY EXIT INIT, {self.score_cb}')

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']

    @sh.utils.call.on_validation
    def after_epoch(self):
        if self.L.np_epoch < self.start_p_epoch: return

        if self.cfg.PARALLEL.IS_MASTER:
            score = self.score_cb.get_score()
            if score > self.max_score:
                self.current_streak = 0
                self.max_score = score
            elif self.current_streak >= self.wait_epoch:
                self.exit_signal += 1
            else:
                if score > 0:
                    self.current_streak += 1
            #suffix = 'ema' if self.cfg.TRAIN.EMA > 0 else 'val'
            # if self.cfg.TRAIN.EMA: self.log_info(f'\t EMA score: {self.score_cb.get_score(ema=True)};')
            # self.log_info(f'\t VAL score: {self.score_cb.get_score(ema=False)};')
            if self.cfg.TRAIN.EMA.ENABLE: self.log_info(f'\t EMA score: {score};')
            self.log_info(f'\t MAX score: {self.max_score}; \n\t Streak: {self.current_streak}')

        if self.cfg.PARALLEL.DDP: torch.distributed.all_reduce(self.exit_signal)
        if self.exit_signal > 0:
            self.log_info('EarlyExit!')
            raise sh.learner.CancelFitException


class FrozenEncoderCB(sh.callbacks.Callback):
    def __init__(self, frozen_sched, logger=None):
        # dict {stage_name : N_EPOCH_UNFREEZE}
        sh.utils.file_op.store_attr(self, locals())
        self.frozen_sched = {k:v for k,v in frozen_sched.items()}
        print(frozen_sched)

    def before_fit(self):
        self.model = sh.utils.nn.unwrap_model(self.L.model)
        stages = self.frozen_sched.keys()
        parts = freeze_stages(self.model, stages, True)
        self.log_warning(f'FREEZING PARTS: {parts}')
        self.enc_frozen = True

    def before_epoch(self):
        e = self.L.n_epoch
        if self.enc_frozen and e in self.frozen_sched.values():
            # stage = [self.frozen_sched[e], ]
            stages = [k for k,v in self.frozen_sched.items() if v == e]
            parts = freeze_stages(self.model, stages, False)
            self.log_warning(f'UNFREEZING PART {parts} at {e}')

            [self.frozen_sched.pop(k) for k in stages]
            if not self.frozen_sched:
                self.enc_frozen = False


class TBPredictionsCB(sh.callbacks.Callback):
    def __init__(self, cfg, writer, batch_read=lambda x: x, denorm=sh.utils.common.denorm, upscale=sh.utils.common.upscale, logger=None):
        sh.utils.file_op.store_attr(self, locals())
        self.num_images, self.hw = 8, (256*3,256)

    def before_fit(self):
        self.mean, self.std = self.cfg.AUGS.MEAN, self.cfg.AUGS.STD

    @torch.no_grad()
    def process_batch(self, training):
        batch = self.batch_read(self.L.batch)
        xb = batch['xb']
        yb = batch['yb']
        # yb = yb[:,1:]
        pr = self.L.pred['yb']
        xb = xb[:self.num_images].cpu().float()
        yb = yb[:self.num_images].cpu().float()
        pr = pr[:self.num_images].cpu().float()

        if pr.shape[1] != 1: # TODO: multilabel
            pr = pr.sum(1, keepdims=True)


        # print(sh.utils.common.st(xb))
        # print(sh.utils.common.st(yb))
        # print(sh.utils.common.st(pr))
        if self.cfg.MODEL.ARCH == 'ssl':
            mask = batch['mask'][:self.num_images].cpu().float()
            mask = mask.unsqueeze(1).repeat(1,3,1,1)
            mask = self.upscale(mask, (xb.shape[2], xb.shape[3]))
            yb = xb.clone()
            yb[mask>0] = 0

        if self.cfg.MODEL.ARCH != 'ssl':
            pr = pr.sigmoid()
            yb = yb.repeat(1,3,1,1)
            yb[:,2] = 0
            yb[:,1] = pr[:,0]
            pr = pr.repeat(1,3,1,1)
            xb = xb * 69  + 176 
            xb = xb / xb.max()
            yb = yb / yb.max()

        xb = self.upscale(xb, self.hw)
        yb = self.upscale(yb, self.hw)
        pr = self.upscale(pr, self.hw)


        # pr = pr.sigmoid()
        b,c,h,w = xb.shape
        x = torch.stack([xb,yb,pr], 2).view(b, 3, h*3, w)
        r = self.upscale(x, self.hw)
        return r

    def process_write_predictions(self, training=False):
        diff = self.process_batch(training=training) # takes last batch that been used
        label = 'train predictions' if training else 'val_predictions'
        diff = torchvision.utils.make_grid(diff, nrow=8, pad_value=4)
        diff = (diff * 255).numpy().astype(np.uint8)
        self.writer.add_image(label+'diff', diff, self.L.n_epoch)
        self.writer.flush()

    def after_epoch(self):
        if self.L.model.training:
            if self.L.n_epoch % self.cfg.TRAIN.TB_STEP == 0:
                self.process_write_predictions(training=True)
        else:
            # try:
            self.process_write_predictions(training=False)
            # except Exception as e:
            #     self.log_error('TB error', e)
            # pred = self.L.pred['seg'].cpu().float().flatten()
            # self.L.writer.add_histogram('predicts/val', pred, self.L.n_epoch)


class CheckpointCB(sh.callbacks.Callback):
    def __init__(self, save_path, logger=None):
        sh.utils.file_op.store_attr(self, locals())

    def _save_dict(self, save_ema):
        save_ema = self.L.model_ema is not None
        model = self.L.model_ema if save_ema else self.L.model
        model_state = sh.utils.nn.get_state_dict(model)

        sd = {
            'epoch': self.L.n_epoch,
            'lr': self.L.lr,
            'model_name': {
                'cls': sh.utils.nn.get_model_name(self.L.model),
            },
            'model_state': {
                'cls': model_state,
            },
            'optim_state': {
                'o': self.L.opt.state_dict(),
            },
            'scaler_state': {
                's': self.L.amp_scaler.state_dict(),
            },
        }
        return sd

    def do_saving(self, name='', save_ema=True):
        self.log_warning(f'Saving new model: {name}')
        torch.save(self._save_dict(save_ema=save_ema), str(self.save_path / f'e{self.L.n_epoch}_t{self.L.total_epochs}_{name}.pth'))



class RegistratorCB(sh.callbacks.Callback):
    def __init__(self, regs, logger=None):
        sh.utils.file_op.store_attr(self, locals())
        self.regs = regs

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']
        for k,v in self.regs.items():
            v = sh.utils.call.get_cb_by_instance(self.L.cbs, v)
            setattr(self.L, k, v)
            self.log_debug(f'Global register callback: {k}: {v}')



class _TrainCallback(sh.callbacks.Callback):
    def batch_acc_ready(self): return self.L.n_batch % self.batch_acc_step == 0

    def get_total_loss(self, loss_d, tracking, ohem=None):
        total_loss = torch.tensor([0.]).cuda()

        for k, loss in loss_d.items():
            if loss is None: continue
            if ohem is not None:
                loss = loss.view(-1)
                run_once(90, self.log_debug, f'{loss.shape, ohem, k}', )
                loss, idxs = torch.topk(loss, int(loss.shape[0]*ohem), largest=False)

            loss = loss.mean()
            if k not in self.loss_weights: self.log_debug(f'There is no weight for {k} loss')
            total_loss += self.loss_weights.get(k, 1) * loss
            if tracking:
                self.L.tracker_cb.set(k + '_train_loss', loss.detach())
        return total_loss

    def clip(self, loss):
        if self.grad_clip_mode != 'none':
            try:
                if self.grad_clip_mode == 'value':
                    torch.nn.utils.clip_grad_norm_(self.L.model.parameters(), self.grad_clip, error_if_nonfinite=True)
                elif self.grad_clip_mode == 'agc':
                    adaptive_clip_grad(self.L.model.parameters(), self.grad_clip)
            except Exception as e:
                self.log_warning('Loss: ', loss)
                # self.log_warning(self._grad_norm())
                self._log_grads(self.L.model)
                self.log_error(f'Invalid grad at clipping {e}')

    def update_step(self, loss, first_pass=True):
        if first_pass and self.sam:
            with self.L.model.no_sync():
                self.amp_scaler.scale(loss).backward()
        else:
            self.amp_scaler.scale(loss).backward()

        if self.batch_acc_ready():
            self.amp_scaler.unscale_(self.L.opt)
            self.clip(loss)

            if self.sam: self.amp_scaler.step(self.L.opt, second=not first_pass)
            else: self.amp_scaler.step(self.L.opt)

            self.amp_scaler.update()
            self.L.opt.zero_grad(set_to_none=True)

    def _log_grads(self, m):
        for i, (tag, param) in enumerate(m.named_parameters()):
            if param.grad is not None and (i == 0 or i == len(list(m.named_parameters()))):
                self.log_debug(i, tag, sh.utils.common.st(param))
                self.log_debug(i, tag, sh.utils.common.st(param.grad))
                break

    def sam_update(self, xb, batch):
        # print('sam', self.sam, xb.shape)
        if self.sam:
            with torch.cuda.amp.autocast(enabled=self.amp_scaler.is_enabled()):
                pred = self.L.model(xb)
                loss_d = self.L.loss_func(pred, batch)
                total_loss = self.get_total_loss(loss_d, tracking=False)
                self.L.pred = pred

            self.update_step(total_loss, first_pass=False)
