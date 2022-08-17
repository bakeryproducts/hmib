from collections import defaultdict
import torch
import hydra
from timm.optim import MADGRAD

import sam
import shallow as sh
import timm.optim.optim_factory as optim_factory


def build_scaler(cfg, init_fn):
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.AMP)
    if cfg.MODEL.INIT_MODEL:
        try:
            init_fn(scaler)
            print('scaler init')
        except Exception as e: print(e)
    return scaler


def build_optim(cfg, model, init_fn):
    if cfg.FEATURES.SAM.RHO > 0:
        # TODO refact this
        # assert not cfg.TRAIN.AMP
        # base_optimizer = torch.optim.SGD
        # base_optimizer = torch.optim.SGD
        base_optimizer = MADGRAD
        # wd = 3e-5
        # momentum = .9
        lr = cfg.OPTIMIZER.LRS[1]
        optimizer = sam.SAM(params=model.parameters(),
                            base_optimizer=base_optimizer,
                            lr=lr,
                            rho=cfg.FEATURES.SAM.RHO,)

    else:
        lr = cfg.OPTIMIZER.LRS[1]

        # weight_decay = None
        # param_groups = optim_factory.add_weight_decay(model, weight_decay)

        #OPTIMIZER:
        #  HEAD_LR:
        #    - name: 'segmentation_head'

        group_names = {k:i for i,k in enumerate([i['name'] for i in cfg.OPTIMIZER.get('FINE_LR', [])])}
        groups = defaultdict(list)
        for name, param in model.named_parameters():
            for gname in group_names:
                if gname in name:
                    groups[gname].append(param)
                    break
            else:
                groups['rest'].append(param)

        options = []
        for group in cfg.OPTIMIZER.get('FINE_LR', []):
            # group  == {'name': 'seg_head', group_params: {'lr_scale': 3, ...}}
            goptions = dict(group.get('group_options', {}))
            gname = group['name']
            goptions["params"] = groups[gname]
            options.append(goptions)

        options.append({'params':groups['rest']})
        optimizer = hydra.utils.instantiate(cfg.OPTIMIZER.OPT, lr=lr)
        optimizer = optimizer(params=options)

    if cfg.MODEL.INIT_MODEL:
        #init_fn(optimizer)
        pass

    return optimizer


class NativeScaler:
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            raise NotImplementedError
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class LrCB(sh.callbacks.Callback):
    def __init__(self, cfg, writer, logger=None):
        sh.utils.file_op.store_attr(self, locals())
        self.sched = self.init_sched(cfg)

    def init_sched(self, cfg):
        _base_scale = 32 * 4
        l0,l1,l2, scale = cfg.OPTIMIZER.LRS
        if scale == 0:
            scale = cfg.TRAIN.BATCH_SIZE / _base_scale / cfg.PARALLEL.WORLD_SIZE

        l0,l1,l2 = l0 * scale, l1 * scale, l2 * scale # scale if for DDP , cfg.PARALLEL.WORLD_SIZE
        warmup_duration = .02
        main_dur = .15

        if cfg.MODEL.INIT_MODEL:
            warmup_duration = .1

        scheds = [
            [warmup_duration,                sh.schedulers.sched_cos(l0,l1)],
            [main_dur,                       sh.schedulers.sched_lin(l1,l1)],
            [1 - warmup_duration - main_dur, sh.schedulers.sched_cos(l1,l2)],
        ]

        sched = sh.schedulers.combine_scheds(scheds)
        # lrcb = sh.callbacks.ParamSchedulerCB('before_epoch', 'lr', sched)
        return sched

    def before_fit(self):
        self.opt = self.L.opt

    @sh.utils.call.on_train
    def before_epoch(self):
        adj_lr = self.sched(self.L.np_epoch)
        self.L.lr = adj_lr

        for param_group in self.opt.param_groups:
            # print(param_group.keys())
            if "lr_scale" in param_group:
                # print(param_group["lr_scale"], adj_lr, len(param_group['params']))
                # pp = param_group['params']
                # for p in pp:
                #     print(p.shape, p.sum())
                param_group["lr"] = adj_lr * param_group["lr_scale"]
            else:
                # print('no scale', adj_lr)
                param_group["lr"] = adj_lr

    # def after_epoch(self):
    #     for i in range(len(self.L.opt.param_groups)):
    #         lr = self.L.opt.param_groups[i]['lr']
    #         d = {v:k for k, v in self.L.opt.GROUP_NAMES.items()}
    #         name = d.get(i, 'rest')
    #         self.L.writer.add_scalar(f'lr/{name}', lr, self.L.n_epoch)
