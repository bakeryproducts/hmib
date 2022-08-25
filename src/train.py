from pathlib import Path
from functools import partial

import torch
import hydra
from torch.utils.tensorboard import SummaryWriter
from fastprogress.fastprogress import master_bar, progress_bar
from omegaconf import OmegaConf

import shallow as sh
from augment import dali
import build_data
import augs
import metrics
import network
import optim
import tv
from logger import logger

from callbacks import CheckpointCB, TrackResultsCB, EarlyExit, TBPredictionsCB, RegistratorCB, ScorerCB, FrozenEncoderCB, get_ema_decay_cb



def start(cfg, output_folder):
    output = None
    if cfg.PARALLEL.IS_MASTER:
        logger.log("INFO", f'Starting on split {cfg.SPLIT}')
        output = output_folder / f'split_{cfg.SPLIT}'
        output.mkdir()

    datasets_generator = build_data.DatasetsGen(cfg)
    datasets = build_data.init_datasets(cfg, datasets_generator, ['TRAIN', 'VALID', 'VALID2'])
    if not cfg.DATA.DALI:
        datasets = augs.create_augmented(cfg, datasets)

    logger.log("DEBUG", 'datasets created.')
    start_split(cfg, output, datasets)
    del datasets
    if cfg.PARALLEL.IS_MASTER: logger.log("DEBUG", f'End of split #{cfg.SPLIT}')


def init_master_cbs(cfg, track_cb, output_folder):
    models_dir = output_folder / 'models'
    tb_dir = output_folder / 'tb'
    models_dir.mkdir()
    tb_dir.mkdir()

    writer = SummaryWriter(log_dir=tb_dir, comment='Demo')
    # TODO fix this
    _s = OmegaConf.to_yaml(cfg).replace(' ', ' &nbsp; ').replace('\n', ' &nbsp; &nbsp;  \n')
    writer.add_text('cfg', _s, 0)

    # _fd = flatten_dict(OmegaConf.to_container(cfg))
    # writer.add_hparams(_fd, {'holder': 0})

    tb_metric_cb = partial(sh.callbacks.TBMetricCB,
                           track_cb=track_cb,
                           writer=writer,
                           train_metrics={
                               'train_losses': [
                                   'seg_dice_train_loss',
                                   'seg_ce_train_loss',
                                   'cls_train_loss',
                                   'ssl_train_loss',
                               ],
                               'general': ['lr', 'train_score']
                           },
                           validation_metrics={
                               'val_losses': [
                                   'seg_dice_ema_loss',
                                   'seg_ce_ema_loss',
                                   'cls_ema_loss',
                                   'ssl_ema_loss',
                               ],
                               'general': ['val_score',
                                           'score',
                                           'ema_dice',
                                           'cls_acc',
                                           ]
                           })

    tb_predict_cb = partial(TBPredictionsCB, cfg=cfg, writer=writer, logger=logger)
    tb_cbs = [tb_metric_cb(), tb_predict_cb()]

    checkpoint_cb = CheckpointCB(models_dir, logger=logger)
    train_timer_cb = sh.callbacks.TimerCB(mode_train=True, logger=logger)
    master_cbs = [train_timer_cb, *tb_cbs, writer, checkpoint_cb, ]
    # master_cbs = [train_timer_cb, *tb_cbs, writer]
    return master_cbs


def init_model_path(p):
    if p:
        p = Path(p)
        if p.suffix == '.pth': return str(p)
        elif p.is_dir():
            p = sh.utils.nn.get_last_model_name(p)
            return str(p)
    return ''


def batch_transform(b):
    xb, yb = b['x'], b['y']
    if isinstance(xb, list):
        xb = torch.vstack(xb)
        yb = torch.vstack(yb)
    return {'xb':xb, 'yb':yb, 'cls':b['cls']}


def batch_transform_dali(b):
    b = b[0]
    xb, yb = b['x'], b['y']
    yb = yb[...,0:1] # RGB -> 1ch
    xb = xb.permute(0,3,1,2)
    yb = yb.permute(0,3,1,2)
    yb = yb.contiguous()
    xb = xb.contiguous()
    return {'xb':xb, 'yb':yb, 'cls':0}


def start_split(cfg, output_folder, datasets):
    n_epochs = cfg.TRAIN.EPOCH_COUNT

    builder = dali.build_dataloaders if cfg.DATA.DALI else build_data.build_dataloaders
    dls = builder(cfg, datasets)
    logger.log("DEBUG", 'build dataloaders.')

    init_path = cfg.MODEL.get("INIT_MODEL", '')
    INIT_MODEL_PATH = init_model_path(init_path)
    model_select = partial(network.model_select, cfg)
    model = model_select()().cuda().train()
    logger.log("DEBUG", 'build model.')

    if cfg.TRAIN.EMA.ENABLE:
        model_ema = sh.utils.ema.ModelEmaV2(model, copy=True).cuda()
        model_ema = model_ema.eval()
        for p in model_ema.parameters():
            p.requires_grad = False
    else:
        model_ema = None

    model = sh.model.wrap_ddp(cfg, model, sync_bn=False, find_unused_parameters=True, broadcast_buffers=True)
    optimizer = optim.build_optim(cfg, model, init_fn=partial(sh.utils.nn.load_state, path=INIT_MODEL_PATH, k=['optim_state', 'o']))
    amp_scaler = optim.build_scaler(cfg, init_fn=partial(sh.utils.nn.load_state, path=INIT_MODEL_PATH, k=['scaler_state', 's']))

    batch_transform_fn = batch_transform_dali if cfg.DATA.DALI else batch_transform
    batch_setup_cb = sh.callbacks.SetupLearnerCB(batch_transform=batch_transform_fn)

    train_cb = tv.TrainCB(amp_scaler=amp_scaler, logger=logger)
    val_cb = tv.ValCB(model_ema=model_ema, logger=logger, batch_transform=batch_transform_fn)

    loss = {}
    for ls in cfg.LOSS:
        loss_fn = hydra.utils.instantiate(ls.LOSS)
        loss[ls.name] = loss_fn


    _crit = partial(metrics.loss_seg, cfg=cfg, loss=loss, reduction='mean')
    criterion = partial(_crit)

    track_cb = TrackResultsCB(logger=logger)
    lr_cb = optim.LrCB(cfg, writer=None, logger=logger)
    score_cb = ScorerCB(logger=logger)
    ema_cb = get_ema_decay_cb(cfg)

    cbs = [batch_setup_cb, lr_cb, ema_cb, train_cb, val_cb]
    if 'UNFREEZE_SCHED' in cfg:
        d = OmegaConf.to_container(cfg.UNFREEZE_SCHED)
        fr_cb = FrozenEncoderCB(d, logger=logger)
        cbs.extend([fr_cb])


    cbs.extend([track_cb, score_cb])
    if cfg.PARALLEL.IS_MASTER: cbs.extend(init_master_cbs(cfg, track_cb, output_folder))
    if cfg.TRAIN.EARLY_EXIT > 0: cbs.extend([EarlyExit(score_cb=score_cb, wait_epoch=cfg.TRAIN.EARLY_EXIT, logger=logger)])
    reg_cb = RegistratorCB(dict(tracker_cb=TrackResultsCB, writer=SummaryWriter), logger=logger)
    cbs.insert(0, reg_cb)

    if cfg.PARALLEL.IS_MASTER:
        epoch_bar = master_bar(range(1, 1 + n_epochs))
        batch_bar = partial(progress_bar, parent=epoch_bar)
    else: epoch_bar, batch_bar = range(1, 1 + n_epochs), lambda x:x

    val_rate = cfg.TRAIN.SCALAR_STEP
    logger.log("WARNING", 'Start learner')
    learner = sh.general_learner.Learner(model, optimizer, sh.utils.file_op.AttrDict(dls), criterion, 0, cbs, batch_bar, epoch_bar, cfg=cfg,)
    learner.fit(n_epochs)
