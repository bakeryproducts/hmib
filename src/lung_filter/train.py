import os
import os.path as osp

import pytorch_lightning as pl

from datamodule import LungFilterDataModule
from module import LungFilterModule
from pl_utils import (
    checkpoint_callback,
    config_args,
    lr_monitor_callback,
    tensorboard_logger,
)


def train(args):
    pl.seed_everything(args.seed)

    datamodule = LungFilterDataModule(**vars(args))
    module = LungFilterModule(**vars(args))

    callbacks = []
    ckpt_callback = checkpoint_callback(args)
    callbacks.append(ckpt_callback)
    callbacks.append(lr_monitor_callback())

    logger = tensorboard_logger(args)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, log_every_n_steps=10)
    trainer.fit(module, datamodule=datamodule)

    print(f"Best model score: {ckpt_callback.best_model_score:.3f}")
    print(f"Best model path: {ckpt_callback.best_model_path}")


def create_if_not_exist(dirpath: str):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)


def main(args):
    create_if_not_exist(args.checkpoints_dir)
    create_if_not_exist(args.log_dir)
    train(args)


if __name__ == '__main__':
    main(config_args(LungFilterModule, LungFilterDataModule))
