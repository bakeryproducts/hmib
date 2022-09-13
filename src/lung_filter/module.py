from argparse import ArgumentParser
from math import ceil

import pytorch_lightning as pl
import timm
import torch
from torch import distributed as dist, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class LungFilterModel(nn.Module):
    def __init__(self):
        super().__init__()


class LungFilterModule(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.mse_criterion = nn.MSELoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()

        self.model = timm.models.create_model(
            self.hparams.model_name, pretrained=True, num_classes=1)

    def forward(self, images):
        return self.model.forward(images)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = self._configure_scheduler(optimizer)
        return [optimizer], [scheduler]

    def _configure_scheduler(self, optimizer):
        dataset_size = len(self.trainer.datamodule.train_dataset)
        step_period = self.hparams.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
        steps_per_epoch = ceil(dataset_size / step_period)
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches < steps_per_epoch:
            steps_per_epoch = self.trainer.limit_train_batches

        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                pct_start=self.hparams.lr_pct_start,
                steps_per_epoch=steps_per_epoch,
                epochs=self.trainer.max_epochs,
            ),
            "interval": "step",
        }

        return scheduler

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model
        parser.add_argument("--model_name", type=str, default="resnet18")
        parser.add_argument("--dropout", type=float, default=0.2)

        # Learning rate
        parser.add_argument("--num_epochs", type=int, default=20)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lr_pct_start", type=float, default=0.1)

        # Monitor
        parser.add_argument("--monitor", type=str, default="losses/val_mse")
        parser.add_argument("--monitor_mode", type=str, default="min")

        return parser

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="val")

    def _step(self, batch, _batch_idx, stage):
        logits = self.forward(batch["image"])[:, 0]
        probs = logits.sigmoid()
        targets = batch["target"]

        # Losses
        bce_loss = self.bce_criterion(logits, targets)
        mse_loss = self.mse_criterion(probs, targets)
        mae_loss = (targets - logits).abs().mean()

        losses = {
            "total": bce_loss + mse_loss + mae_loss,
            "bce": bce_loss,
            "mse": mse_loss,
            "mae": mae_loss,
            "avg_target": targets.mean()
        }

        metrics = dict()
        self._log(losses, metrics, stage=stage)

        if stage == "train":
            return losses["total"]

        return {
            "probabilities": probs.cpu(),
        }

    def _log(self, losses, metrics, stage):
        # Progress bar
        progress_bar = dict()
        if stage != "train":
            progress_bar[f"{stage}_loss"] = losses["total"]
        self.log_dict(progress_bar, prog_bar=True, logger=False)

        # Logs
        logs = dict()
        for lname, lval in losses.items():
            logs[f"losses/{stage}_{lname}"] = lval
        self.log_dict(logs, prog_bar=False, logger=True)

    def _gather(self, key, outs):
        node_values = torch.concat([torch.as_tensor(output[key]).to(self.device) for output in outs])
        if self.trainer.world_size == 1:
            return node_values

        all_values = [torch.zeros_like(node_values) for _ in range(self.trainer.world_size)]
        dist.barrier()
        dist.all_gather(all_values, node_values)
        all_values = torch.cat(all_values)
        return
