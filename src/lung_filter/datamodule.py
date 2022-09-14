import os.path as osp
from argparse import ArgumentParser

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from dataset import LungFilterDataset


class LungFilterDataModule(pl.LightningDataModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        size = self.hparams.input_size

        self.pre_transforms = []
        self.augmentations = [
            A.RandomResizedCrop(size, size, scale=(0.33, 3.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, hue=0.4, saturation=0.4, p=0.7),
        ]

        self.post_transforms = [
            A.Resize(size, size, always_apply=True),
            A.Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # DataModule already configured
        if stage == "test":
            return

        items = LungFilterDataset.load_items(
            images_dir=self.hparams.images_dir,
            cache=self.hparams.cache,
            debug=self.hparams.debug)

        splits = pd.read_csv(osp.join(self.hparams.images_dir, "splits.csv"))
        splits = {str(row.image_name): row.split for row in splits.itertuples()}
        train_items = [item for item in items if splits[item["image_name"]] == "train"]
        val_items = [item for item in items if splits[item["image_name"]] == "val"]

        # Train dataset
        train_transform = A.Compose(self.pre_transforms + self.augmentations + self.post_transforms)
        self.train_dataset = LungFilterDataset(train_items, train_transform, rate=self.hparams.rate)

        # Val dataset
        val_transform = A.Compose(self.pre_transforms + self.post_transforms)
        self.val_dataset = LungFilterDataset(val_items, val_transform)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Paths
        parser.add_argument("--images_dir", type=str, default="../../input/hmib/lung_filter")

        # General
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--input_size", type=int, default=224)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--val_size", type=int, default=0.2)
        parser.add_argument("--cache", action="store_true")
        parser.add_argument("--rate", type=int, default=100)

        return parser

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, batch_sampler=None, shuffle=False, batch_size=None, drop_last=False):
        if batch_size is None:
            batch_size = self.hparams.batch_size

        params = {
            "pin_memory": True,
            "persistent_workers": self.hparams.num_workers > 0,
            "num_workers": self.hparams.num_workers,
        }

        if batch_sampler is not None:
            params["batch_sampler"] = batch_sampler
        else:
            params["batch_size"] = batch_size
            params["shuffle"] = shuffle
            params["drop_last"] = drop_last

        return DataLoader(dataset, **params)
